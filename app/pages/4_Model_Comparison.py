"""Model Comparison - real held-out test metrics, charts and confusion matrices."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import MODEL_COLORS, PROXY_NOTE, current_threshold, page_setup, read_csv, require

page_setup("Model Comparison", "📈")
st.title("📈 Model Comparison")
if not require(["results/metrics/test_metrics_table.csv", "results/metrics/test_predictions.csv"],
               "Run `python -m src.training.evaluate` first."):
    st.stop()

from src.prediction.irrigation import evaluate_decisions  # noqa: E402

tbl = read_csv("results/metrics/test_metrics_table.csv", index_col=0)
pred = read_csv("results/metrics/test_predictions.csv", parse_dates=["target_time"])
models = [c for c in pred.columns if c not in ("target_time", "actual")]

st.markdown(f"Held-out chronological test set: **{len(pred):,}** next-hour targets from "
            f"{pred.target_time.min():%Y-%m-%d} to {pred.target_time.max():%Y-%m-%d} (UTC). "
            "Units: m³/m³. *Persistence* = naive baseline that repeats the last observed value.")
st.dataframe(tbl.style.format("{:.6f}").highlight_min(subset=["MAE", "MSE", "RMSE"], color="#e8f6ef")
             .highlight_max(subset=["R2"], color="#e8f6ef"), use_container_width=True)
st.caption("RMSE_skill_vs_persistence = 1 − RMSE_model / RMSE_persistence (positive = better than persistence).")

cols = st.columns(4)
for col, met in zip(cols, ["MAE", "RMSE", "MSE", "R2"]):
    fig = px.bar(tbl.reset_index(), x="model", y=met, color="model", color_discrete_map=MODEL_COLORS,
                 template="plotly_white", text_auto=".4g")
    fig.update_layout(showlegend=False, height=300, margin=dict(t=30, b=10), title=met)
    if met == "R2":
        fig.update_yaxes(range=[max(0, tbl[met].min() - 0.02), 1])
    col.plotly_chart(fig, use_container_width=True)

st.subheader("Actual vs predicted")
show = st.multiselect("Series", models, default=models)
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred.target_time, y=pred.actual, name="Actual", line=dict(color="black", width=1.6)))
for m in show:
    fig.add_trace(go.Scatter(x=pred.target_time, y=pred[m], name=m, line=dict(color=MODEL_COLORS.get(m), width=1)))
thr = current_threshold()
fig.add_hline(y=thr["threshold"], line_dash="dash", line_color="#b42318", annotation_text="threshold")
fig.update_layout(height=420, template="plotly_white", yaxis_title="VWC (m³/m³)",
                  xaxis=dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
m_sel = c1.selectbox("Scatter for model", models, index=len(models) - 1)
fig = px.scatter(pred, x="actual", y=m_sel, opacity=0.4, template="plotly_white",
                 color_discrete_sequence=[MODEL_COLORS.get(m_sel)])
lo, hi = pred.actual.min(), pred.actual.max()
fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="black"), name="y = x"))
fig.update_layout(height=380)
c1.plotly_chart(fig, use_container_width=True)
err = pd.DataFrame({m: pred[m] - pred.actual for m in models}).melt(var_name="model", value_name="error")
fig = px.histogram(err, x="error", color="model", barmode="overlay", nbins=120, opacity=0.5,
                   color_discrete_map=MODEL_COLORS, template="plotly_white")
fig.update_layout(height=380, title="Error distribution (predicted − actual)")
c2.plotly_chart(fig, use_container_width=True)

st.subheader("Irrigation decisions (threshold from Settings)")
st.caption(f"Threshold {thr['threshold']:.4f} m³/m³ — {thr['source']}. {PROXY_NOTE}")
res = {m: evaluate_decisions(pred[m].to_numpy(), pred.actual.to_numpy(), thr["threshold"]) for m in models}
summary = pd.DataFrame({m: {k: r[k] for k in ("accuracy", "precision_irrigate", "recall_irrigate",
                                               "f1_irrigate", "f1_macro", "positive_rate_actual")}
                        for m, r in res.items()}).T
st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)
cols = st.columns(len(models))
for col, m in zip(cols, models):
    cm = res[m]["confusion_matrix"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["NO", "YES"], y=["NO", "YES"],
                    labels=dict(x="Predicted", y="Proxy (observed)"))
    fig.update_layout(title=f"{m}<br><sup>recall YES {res[m]['recall_irrigate']:.3f}</sup>", height=300,
                      coloraxis_showscale=False, margin=dict(t=60, b=10))
    col.plotly_chart(fig, use_container_width=True)

sweep = read_csv("results/metrics/irrigation_threshold_sweep.csv")
if sweep is not None:
    st.subheader("Threshold sensitivity (training-percentile thresholds)")
    fig = px.line(sweep, x="threshold", y="recall_irrigate", color="model", markers=True,
                  color_discrete_map=MODEL_COLORS, template="plotly_white", hover_data=["train_percentile", "f1_irrigate"])
    fig.update_layout(height=340, yaxis_title="Recall (IRRIGATE)")
    st.plotly_chart(fig, use_container_width=True)
