"""Unit and integrity tests. Run: python -m pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.create_sequences import make_windows, split_timeline
from src.data.inspect import sentinel_for
from src.data.preprocess import contiguous_segments, interpolate_short_gaps
from src.models import MODEL_NAMES, build_model
from src.prediction.irrigation import decide, evaluate_decisions, recommendation, resolve_threshold
from src.utils import load_config, project_path

CFG = load_config()


# ---------- parsing / cleaning ----------
def test_sentinels_match_noaa_readme():
    # README Note C: -9999.0 (7 chars, 1 dp), -99.000 (7 chars, 3 dp)
    assert sentinel_for(7, 1) == -9999.0
    assert sentinel_for(7, 3) == -99.0
    assert sentinel_for(6, 0) == -99999.0
    assert sentinel_for(5, 0) == -9999.0


def test_interpolation_only_fills_short_interior_gaps():
    idx = pd.date_range("2020-01-01", periods=20, freq="h")
    s = pd.Series(np.arange(20, dtype=float), index=idx)
    s.iloc[2:5] = np.nan     # 3-h gap -> filled
    s.iloc[8:15] = np.nan    # 7-h gap -> untouched
    s.iloc[-1] = np.nan      # trailing -> untouched (not interior)
    out, n = interpolate_short_gaps(s, 6)
    assert n == 3
    assert np.allclose(out.iloc[2:5], [2, 3, 4])
    assert out.iloc[8:15].isna().all()
    assert np.isnan(out.iloc[-1])


def test_contiguous_segments_break_on_gaps():
    idx = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 03:00", "2020-01-01 04:00"])
    assert contiguous_segments(idx).tolist() == [0, 0, 1, 1]


# ---------- windowing / leakage ----------
def _toy_frame(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "SM": np.arange(n, dtype=float) * 10}, index=idx)
    df["segment_id"] = 0
    return df


def test_window_target_is_next_hour():
    df = _toy_frame(40)
    df["seg"] = "0"
    X, y, t, last = make_windows(df, ["a", "SM"], "SM", window=24, horizon=1, stride=1)
    assert X.shape == (16, 24, 2)
    assert y[0] == 240.0 and last[0] == 230.0
    assert pd.Timestamp(t[0]) == df.index[24]


def test_windows_do_not_cross_segment_gaps():
    df = _toy_frame(60)
    df = df.drop(df.index[30])                # gap
    df["seg"] = contiguous_segments(df.index).astype(str)
    X, y, t, _ = make_windows(df, ["a", "SM"], "SM", 24, 1, 1)
    for xi, ti in zip(X, pd.to_datetime(t)):
        assert np.all(np.diff(xi[:, 0]) == 1)  # consecutive hours within each window
    assert len(y) == (30 - 24) + (29 - 24)


def test_split_is_chronological():
    df = _toy_frame(1000)
    split, _ = split_timeline(df, 24, 1, 0.7, 0.15)
    order = split.map({"train": 0, "val": 1, "test": 2}).to_numpy()
    assert np.all(np.diff(order) >= 0)


@pytest.mark.skipif(not project_path("data/processed/sequence_metadata.json").exists(), reason="pipeline not run")
def test_saved_splits_have_no_temporal_overlap():
    import json

    meta = json.loads(project_path("data/processed/sequence_metadata.json").read_text())
    s = meta["splits"]
    w = meta["input_window"]
    assert pd.Timestamp(s["train"]["target_time_end"]) < pd.Timestamp(s["val"]["target_time_start"]) - pd.Timedelta(hours=w)
    assert pd.Timestamp(s["val"]["target_time_end"]) < pd.Timestamp(s["test"]["target_time_start"]) - pd.Timedelta(hours=w)
    for k in ("train", "val", "test"):
        d = np.load(project_path(f"data/processed/windows_{k}.npz"))
        assert d["X"].shape[1:] == (24, len(CFG["features"]))
        assert np.isfinite(d["X"]).all() and np.isfinite(d["y"]).all()
        assert np.all(np.diff(d["target_time"].astype("int64")) > 0)  # strictly increasing, unshuffled


# ---------- models ----------
@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_model_output_shape(name):
    m = build_model(name, 11, CFG["models"][name])
    out = m(torch.randn(5, 24, 11))
    assert out.shape == (5,)


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_eval_mode_is_deterministic_and_uses_latest_hour(name):
    torch.manual_seed(0)
    m = build_model(name, 11, CFG["models"][name]).eval()
    x = torch.randn(2, 24, 11)
    assert torch.equal(m(x), m(x))                 # dropout disabled in eval
    x2 = x.clone(); x2[:, -1] += 3.0                # perturb most recent hour
    assert not torch.allclose(m(x), m(x2))


# ---------- irrigation ----------
def test_decision_rule():
    assert decide(0.20, 0.25) and not decide(0.30, 0.25)
    r = recommendation(0.2, 0.25)
    assert r["irrigate"] == "YES" and r["margin"] < 0


def test_threshold_from_train_percentile():
    y = np.linspace(0, 1, 101)
    t = resolve_threshold({"mode": "train_percentile", "percentile": 25}, y)
    assert t["threshold"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        resolve_threshold({"mode": "fixed", "threshold": None})


def test_decision_metrics():
    actual = np.array([0.1, 0.1, 0.4, 0.4])
    pred = np.array([0.1, 0.4, 0.4, 0.1])
    r = evaluate_decisions(pred, actual, 0.25)
    assert r["confusion_matrix"] == [[1, 1], [1, 1]]
    assert r["recall_irrigate"] == 0.5 and r["accuracy"] == 0.5


def test_delta_mode_reconstructs_level():
    from sklearn.preprocessing import StandardScaler

    from src.training.evaluate import predict_level

    class Zero(torch.nn.Module):          # network predicting zero scaled change
        def forward(self, x):
            return torch.zeros(x.shape[0])

    raw = np.random.default_rng(0).normal(0.3, 0.05, size=(50, 2))
    xs = StandardScaler().fit(raw)
    ys = StandardScaler().fit(np.zeros((10, 1)) + np.arange(10)[:, None] * 0.001)
    X = xs.transform(raw).reshape(2, 25, 2)[:, :24].astype(np.float32)
    sc = {"x_scaler": xs, "y_scaler": ys, "target_mode": "delta", "target_index": 1}
    out = predict_level(Zero(), X, sc)
    expected = xs.inverse_transform(X[:, -1, :])[:, 1] + ys.mean_[0]
    assert np.allclose(out, expected, atol=1e-5)
