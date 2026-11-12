"""Model Training - retrain LSTM / BiLSTM / Transformer with live progress."""
from __future__ import annotations

import copy

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import MODEL_NAMES, config, page_setup, read_csv, read_json, require

page_setup("Model Training", "🧠")
st.title("🧠 Model Training")
if not require(["data/processed/windows_train.npz"], "Run `python -m src.data.create_sequences` first."):
    st.stop()

cfg = config()
seq = read_json("data/processed/sequence_metadata.json")
st.markdown(f"Training windows: **{seq['splits']['train']['n_windows']:,}** · validation: "
            f"**{seq['splits']['val']['n_windows']:,}** · input {seq['input_window']} h × "
            f"{len(seq['features'])} features → next-hour `{seq['target']}`. "
            "Protocol: MSELoss, Adam, early stopping on validation loss, seed "
            f"{cfg['project']['seed']}.")

st.subheader("Saved models")
rows = []
for k, d in MODEL_NAMES.items():
    mc = read_json(f"models/{k}_config.json")
    rows.append({"model": d, "trained": mc is not None,
                 **({"best_epoch": mc["best_epoch"], "epochs_run": mc["epochs_run"],
                     "params": mc["n_trainable_params"], "best_val_mse_scaled": round(mc["best_val_mse_scaled"], 5),
                     "time_s": mc["training_time_s"], "trained_at": mc["trained_at"]} if mc else {})})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

hist_cols = st.columns(3)
for col, (k, d) in zip(hist_cols, MODEL_NAMES.items()):
    h = read_csv(f"results/metrics/{k}_history.csv")
    if h is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.epoch, y=h.train_loss, name="train"))
        fig.add_trace(go.Scatter(x=h.epoch, y=h.val_loss, name="validation", line=dict(dash="dash")))
        fig.update_layout(title=d, height=280, template="plotly_white", yaxis_type="log",
                          margin=dict(t=40, b=10), legend=dict(orientation="h"))
        col.plotly_chart(fig, use_container_width=True)

st.subheader("Train / retrain")
st.warning("Retraining overwrites the saved weights in `models/`. Re-run evaluation afterwards so the "
           "comparison and explainability pages reflect the new weights.")
with st.form("train"):
    c1, c2, c3 = st.columns(3)
    chosen = c1.multiselect("Models", list(MODEL_NAMES), default=list(MODEL_NAMES), format_func=MODEL_NAMES.get)
    epochs = c2.number_input("Max epochs", 1, 300, cfg["training"]["epochs"])
    lr = c3.number_input("Learning rate", 1e-5, 1e-1, cfg["training"]["learning_rate"], format="%.5f")
    c4, c5, c6 = st.columns(3)
    patience = c4.number_input("Early-stopping patience", 1, 50, cfg["training"]["early_stopping_patience"])
    batch = c5.selectbox("Batch size", [64, 128, 256, 512], index=[64, 128, 256, 512].index(cfg["training"]["batch_size"]))
    run_eval = c6.checkbox("Run evaluation after training", value=True)
    go_btn = st.form_submit_button("Start training", type="primary")

if go_btn and chosen:
    from src.training.train import train_model

    run_cfg = copy.deepcopy(cfg)
    run_cfg["training"].update(learning_rate=float(lr), early_stopping_patience=int(patience), batch_size=int(batch))
    for k in chosen:
        st.markdown(f"**{MODEL_NAMES[k]}**")
        bar = st.progress(0.0, text="starting…")
        chart = st.empty()
        recs = []

        def cb(r, bar=bar, chart=chart, recs=recs):
            recs.append(r)
            bar.progress(r["epoch"] / r["epochs"],
                         text=f"epoch {r['epoch']}/{r['epochs']} · train {r['train_loss']:.4f} · val {r['val_loss']:.4f}")
            h = pd.DataFrame(recs)
            chart.line_chart(h.set_index("epoch")[["train_loss", "val_loss"]], height=220)

        summary = train_model(k, run_cfg, epochs=int(epochs), progress_callback=cb)
        bar.progress(1.0, text=f"done · best epoch {summary['best_epoch']} · "
                               f"val MSE {summary['best_val_mse_scaled']:.5f} · {summary['training_time_s']} s")
    if run_eval:
        with st.spinner("Evaluating on the held-out test set…"):
            from src.training import evaluate

            evaluate.main()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Training finished. Models and metrics updated.")
