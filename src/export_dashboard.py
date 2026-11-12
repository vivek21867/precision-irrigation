"""Export the data behind the web dashboard (web/) to one JSON file.

Every value is read from real pipeline outputs (processed NOAA data, trained
weights, results/metrics). Nothing is hard-coded. Run after evaluation:

    python -m src.export_dashboard            # -> web/public/data/dashboard.json
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import MODEL_NAMES
from src.prediction.manual import ManualPredictor
from src.utils import exp_paths, get_logger, load_config, project_path

log = get_logger("export_dashboard", "export_dashboard.log")

HISTORY_DAYS = 7
TEST_TAIL_HOURS = 14 * 24
CONDITIONS = ["T_HR_AVG", "RH_HR_AVG", "P_CALC", "SOLARAD", "SUR_TEMP", "SOIL_TEMP_20"]


def _clean(v):
    """JSON-safe scalars (NaN -> None, numpy -> python, rounded to 6 dp)."""
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else round(float(v), 6)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _ts(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M")


def _default_model(metrics: pd.DataFrame) -> str:
    """Model chosen in the Streamlit Settings page, else the lowest test RMSE network."""
    settings = project_path("app_settings.json")
    if settings.exists():
        chosen = MODEL_NAMES.get(json.loads(settings.read_text("utf-8")).get("model", ""))
        if chosen:
            return chosen
    nets = metrics[metrics["model"] != "Persistence"]
    return str(nets.loc[nets["RMSE"].idxmin(), "model"])


def build(cfg: dict) -> dict:
    paths = exp_paths(cfg)
    metrics_dir = paths["metrics"]
    features, target = cfg["features"], cfg["target"]
    win = cfg["sequences"]["input_window"]

    df = pd.read_parquet(project_path(cfg["data"]["processed_dir"]) / "cleaned_hourly.parquet")
    irr = json.loads((metrics_dir / "irrigation_metrics.json").read_text(encoding="utf-8"))
    threshold = irr["threshold"]["threshold"]

    # Latest contiguous 24-h window -> next-hour forecast from every trained model
    # (same code path as the Manual Predict API).
    last = df.iloc[-win:]
    if last["segment_id"].nunique() != 1:
        raise RuntimeError("Last 24 rows are not one contiguous hourly window")
    latest_time = df.index[-1]
    latest = df.iloc[-1]
    predictor = ManualPredictor(cfg)
    result = predictor.predict(last[features].to_numpy())
    forecasts = {name: p["value"] for name, p in result["predictions"].items()}
    model_cfgs = {}
    for key, label in MODEL_NAMES.items():
        p = paths["models"] / f"{key}_config.json"
        if p.exists():
            c = json.loads(p.read_text("utf-8"))
            model_cfgs[label] = {k: c.get(k) for k in ("hyperparameters", "n_trainable_params", "epochs_run", "best_epoch")}

    hist = df.loc[df.index > latest_time - pd.Timedelta(days=HISTORY_DAYS)]
    history = [{"t": _ts(t), **{f: _clean(r[f]) for f in features}} for t, r in hist.iterrows()]

    tp = pd.read_csv(metrics_dir / "test_predictions.csv").tail(TEST_TAIL_HOURS)
    tp["target_time"] = pd.to_datetime(tp["target_time"]).map(_ts)

    metrics = pd.read_csv(metrics_dir / "test_metrics_table.csv")
    perm = pd.read_csv(metrics_dir / "permutation_importance.csv")
    shap = pd.read_csv(metrics_dir / "shap_mean_abs_by_feature.csv", index_col=0)
    sweep = pd.read_csv(metrics_dir / "irrigation_threshold_sweep.csv")
    histories = {}
    for key, label in MODEL_NAMES.items():
        p = metrics_dir / f"{key}_history.csv"
        if p.exists():
            histories[label] = _records(pd.read_csv(p)[["epoch", "train_loss", "val_loss"]])

    seq = json.loads((paths["windows"] / "sequence_metadata.json").read_text("utf-8"))
    prep = json.loads((paths["reports"] / "preprocessing_report.json").read_text("utf-8"))

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "station": {"name": cfg["data"]["station"], "source": "NOAA NCEI U.S. Climate Reference Network, hourly02",
                    "url": cfg["data"]["base_url"]},
        "target": target,
        "target_mode": seq["target_mode"],
        "default_model": _default_model(metrics),
        "threshold": {"value": _clean(threshold), "source": irr["threshold"]["source"]},
        "latest": {"time": _ts(latest_time), "lst_time": _ts(latest["LST_DATETIME"]),
                   "values": {f: _clean(latest[f]) for f in features},
                   "forecast_time": _ts(latest_time + pd.Timedelta(hours=1))},
        "forecasts": {k: _clean(v) for k, v in forecasts.items()},
        "input_window": [{"t": _ts(t), "v": _clean(v)} for t, v in last[target].items()],
        "history": history,
        "conditions": CONDITIONS,
        "model_io": {"features": predictor.feature_info(), "input_window": win,
                     "horizon": cfg["sequences"]["horizon"], "models": model_cfgs},
        "metrics": _records(metrics),
        "irrigation": {m: {k: (v if k == "confusion_matrix" else _clean(v)) for k, v in d.items()}
                       for m, d in irr["models"].items()},
        "threshold_sweep": _records(sweep),
        "test_predictions": _records(tp),
        "permutation_importance": _records(perm),
        "shap_by_feature": {m: {f: _clean(v) for f, v in shap[m].items()} for m in shap.columns},
        "training_history": histories,
        "dataset": {"raw_rows": prep["n_input_rows"], "clean_rows": prep["n_clean_rows"],
                    "segments": prep["n_segments"], "start": prep["clean_utc_start"], "end": prep["clean_utc_end"],
                    "rows_by_year": prep["clean_rows_by_year"],
                    "splits": {k: {"windows": v["n_windows"], "start": v["target_time_start"],
                                   "end": v["target_time_end"], "mean": _clean(v["target_mean"])}
                               for k, v in seq["splits"].items()}},
    }


def main(out: str = "web/public/data/dashboard.json") -> int:
    cfg = load_config()
    data = build(cfg)
    path = project_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    log.info("Wrote %s (%.0f kB): forecasts %s", path, path.stat().st_size / 1024, data["forecasts"])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
