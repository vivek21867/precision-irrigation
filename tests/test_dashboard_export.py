"""The web dashboard JSON must be a faithful copy of real pipeline outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
READY = (ROOT / "models/transformer.pt").exists() and (ROOT / "results/metrics/test_metrics_table.csv").exists()


@pytest.mark.skipif(not READY, reason="models/results not generated yet")
def test_export_matches_pipeline_outputs():
    from src.export_dashboard import build
    from src.utils import load_config

    data = json.loads(json.dumps(build(load_config())))  # must be JSON-serialisable
    metrics = pd.read_csv(ROOT / "results/metrics/test_metrics_table.csv")
    for row in data["metrics"]:
        ref = metrics.set_index("model").loc[row["model"]]
        assert row["RMSE"] == pytest.approx(ref["RMSE"], abs=1e-6)

    irr = json.loads((ROOT / "results/metrics/irrigation_metrics.json").read_text("utf-8"))
    assert data["threshold"]["value"] == pytest.approx(irr["threshold"]["threshold"], abs=1e-6)
    assert data["irrigation"]["Transformer"]["confusion_matrix"] == irr["models"]["Transformer"]["confusion_matrix"]

    clean = pd.read_parquet(ROOT / "data/processed/cleaned_hourly.parquet")
    target = data["target"]
    assert data["forecasts"]["Persistence"] == pytest.approx(float(clean[target].iloc[-1]), abs=1e-6)
    assert [p["v"] for p in data["input_window"]] == pytest.approx(clean[target].iloc[-24:].to_numpy(), abs=1e-6)
    assert all(np.isfinite(v) for v in data["forecasts"].values())
    assert data["history"][-1]["t"] == data["latest"]["time"]
