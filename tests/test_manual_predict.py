"""Manual-input prediction path used by the web Predict page (src/prediction/manual.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
READY = (ROOT / "models/transformer.pt").exists() and (ROOT / "results/metrics/irrigation_metrics.json").exists()
pytestmark = pytest.mark.skipif(not READY, reason="models/results not generated yet")


@pytest.fixture(scope="module")
def predictor():
    from src.prediction.manual import ManualPredictor
    from src.utils import load_config

    return ManualPredictor(load_config())


def test_latest_window_matches_offline_forecast(predictor):
    from src.training.evaluate import predict_level

    rows = predictor.meta()["latest_window"]["rows"]
    res = predictor.predict(rows)
    X = predictor.sc["x_scaler"].transform(np.asarray(rows)).reshape(1, 24, -1).astype(np.float32)
    for name, model in predictor.models.items():
        assert res["predictions"][name]["value"] == pytest.approx(float(predict_level(model, X, predictor.sc)[0]), abs=1e-5)
    assert res["predictions"]["Persistence"]["value"] == pytest.approx(rows[-1][predictor.features.index(predictor.target)])


def test_decision_follows_threshold(predictor):
    res = predictor.predict(predictor.meta()["latest_window"]["rows"])
    thr = res["threshold"]["threshold"]
    for p in res["predictions"].values():
        assert p["irrigate"] == ("YES" if p["value"] < thr else "NO")


def test_rejects_bad_shape_and_physical_bounds(predictor):
    rows = [r[:] for r in predictor.meta()["latest_window"]["rows"]]
    with pytest.raises(ValueError, match="Expected 24 hours"):
        predictor.predict(rows[:5])
    rows[-1][predictor.features.index("RH_HR_AVG")] = 150
    with pytest.raises(ValueError, match="Relative humidity"):
        predictor.predict(rows)
    rows[-1][predictor.features.index("RH_HR_AVG")] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(rows)


def test_warns_outside_training_range(predictor):
    rows = [r[:] for r in predictor.meta()["latest_window"]["rows"]]
    rows[-1][predictor.features.index("P_CALC")] = predictor.train_range["P_CALC"][1] + 10
    assert any("Precipitation" in w for w in predictor.predict(rows)["warnings"])
