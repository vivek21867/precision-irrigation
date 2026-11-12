"""Chronological split + sliding-window dataset creation.

Leakage controls:
  * The cleaned TIMELINE is split first (70/15/15 of usable hours, by time),
    then windows are built inside each split separately, so no window spans a
    split boundary.
  * Windows are built only within contiguous hourly segments (no gaps).
  * StandardScalers are fitted on training-split rows only.
  * Nothing is shuffled here; the training loader may shuffle *windows* of the
    training split only (each window keeps its internal time order).

Window definition: inputs = hours t-23..t (24 h), target = TARGET at t+1.
With sequences.target_mode = "delta" the network's training target is
TARGET(t+1) - TARGET(t); the level TARGET(t+1) is always kept as `y_raw`.

Usage:
    python -m src.data.create_sequences
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import exp_paths, get_logger, load_config, project_path

log = get_logger("sequences", "sequences.log")
SPLITS = ("train", "val", "test")


def split_timeline(clean: pd.DataFrame, window: int, horizon: int,
                   train_frac: float, val_frac: float) -> tuple[pd.Series, dict]:
    """Assign each row to train/val/test by chronological cut timestamps.

    Cuts are placed at the train_frac / train_frac+val_frac quantiles of rows that
    belong to segments long enough to yield at least one window.
    """
    seg_len = clean.groupby("segment_id")["segment_id"].transform("size")
    usable = clean.index[seg_len >= window + horizon]
    n = len(usable)
    t1 = usable[int(n * train_frac)]
    t2 = usable[int(n * (train_frac + val_frac))]
    split = pd.Series(np.where(clean.index < t1, "train", np.where(clean.index < t2, "val", "test")),
                      index=clean.index)
    return split, {"train_end_exclusive": str(t1), "val_end_exclusive": str(t2)}


def make_windows(df: pd.DataFrame, features: list[str], target: str, window: int,
                 horizon: int, stride: int, seg_col: str = "seg") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, target_time, last_input_target) windows within each contiguous segment."""
    Xs, ys, ts, lasts = [], [], [], []
    tgt_idx = features.index(target)
    # sort=False keeps segments in order of first appearance (chronological, since df is time-sorted);
    # sort=True would order string ids lexicographically ("10_train" < "2_train").
    for _, g in df.groupby(seg_col, sort=False):
        vals = g[features].to_numpy(dtype=np.float32)
        tvals = g[target].to_numpy(dtype=np.float32)
        n = len(g)
        for start in range(0, n - window - horizon + 1, stride):
            end = start + window
            Xs.append(vals[start:end])
            ys.append(tvals[end + horizon - 1])
            ts.append(g.index[end + horizon - 1])
            lasts.append(vals[end - 1, tgt_idx])
    if not Xs:
        f = len(features)
        return (np.empty((0, window, f), np.float32), np.empty(0, np.float32),
                np.empty(0, "datetime64[ns]"), np.empty(0, np.float32))
    return (np.stack(Xs), np.asarray(ys, np.float32),
            np.asarray(pd.DatetimeIndex(ts).as_unit("ns").values), np.asarray(lasts, np.float32))


def main() -> int:
    cfg = load_config()
    s = cfg["sequences"]
    features, target = cfg["features"], cfg["target"]
    mode = s.get("target_mode", "level")
    if mode not in ("level", "delta"):
        raise ValueError(f"Unknown target_mode {mode}")
    out = exp_paths(cfg)["windows"]
    clean = pd.read_parquet(project_path(cfg["data"]["processed_dir"]) / "cleaned_hourly.parquet")
    clean = clean.sort_index()
    assert clean.index.is_monotonic_increasing and not clean.index.has_duplicates

    split, cuts = split_timeline(clean, s["input_window"], s["horizon"], s["train_frac"], s["val_frac"])
    clean["split"] = split
    # Segment ids restart within each split so windows cannot cross split boundaries.
    clean["seg"] = clean["segment_id"].astype(str) + "_" + clean["split"]

    built = {name: make_windows(clean[clean["split"] == name], features, target,
                                s["input_window"], s["horizon"], s["stride"]) for name in SPLITS}

    def model_target(y, last):
        return y - last if mode == "delta" else y

    # Scalers fitted on TRAINING data only.
    train_rows = clean[clean["split"] == "train"]
    x_scaler = StandardScaler().fit(train_rows[features].to_numpy())
    _, y_tr, _, last_tr = built["train"]
    y_scaler = StandardScaler().fit(model_target(y_tr, last_tr).reshape(-1, 1))

    meta = {"features": features, "target": target, "target_mode": mode, "input_window": s["input_window"],
            "horizon": s["horizon"], "stride": s["stride"], "cuts": cuts, "splits": {}}
    for name in SPLITS:
        part = clean[clean["split"] == name]
        X, y, t, last = built[name]
        Xs = x_scaler.transform(X.reshape(-1, len(features))).reshape(X.shape).astype(np.float32)
        ys = y_scaler.transform(model_target(y, last).reshape(-1, 1)).ravel().astype(np.float32)
        np.savez_compressed(out / f"windows_{name}.npz", X=Xs, y=ys, y_raw=y, target_time=t,
                            last_observed_target=last)
        meta["splits"][name] = {
            "rows": int(len(part)),
            "row_time_start": str(part.index.min()), "row_time_end": str(part.index.max()),
            "n_windows": int(len(y)), "X_shape": list(Xs.shape), "y_shape": list(ys.shape),
            "target_time_start": str(pd.Timestamp(t.min())) if len(t) else None,
            "target_time_end": str(pd.Timestamp(t.max())) if len(t) else None,
            "target_mean": float(y.mean()) if len(y) else None,
            "target_std": float(y.std()) if len(y) else None,
        }
        log.info("%-5s rows=%d windows=%d X=%s  targets %s -> %s", name, len(part), len(y), Xs.shape,
                 meta["splits"][name]["target_time_start"], meta["splits"][name]["target_time_end"])

    # Leakage checks
    tr_end = pd.Timestamp(meta["splits"]["train"]["target_time_end"])
    va_start = pd.Timestamp(meta["splits"]["val"]["target_time_start"])
    va_end = pd.Timestamp(meta["splits"]["val"]["target_time_end"])
    te_start = pd.Timestamp(meta["splits"]["test"]["target_time_start"])
    assert tr_end < pd.Timestamp(cuts["train_end_exclusive"]) <= va_start - pd.Timedelta(hours=s["input_window"])
    assert va_end < pd.Timestamp(cuts["val_end_exclusive"]) <= te_start - pd.Timedelta(hours=s["input_window"])
    for name in SPLITS:
        t = np.load(out / f"windows_{name}.npz")["target_time"]
        assert np.all(np.diff(t.astype("int64")) > 0), f"{name} windows not in chronological order"
    meta["leakage_checks"] = ("passed: every window's inputs and target lie within its own split; "
                              "windows stored in strictly increasing target time")

    with open(out / "scalers.pkl", "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "features": features, "target": target,
                     "target_mode": mode, "target_index": features.index(target)}, f)
    with open(out / "sequence_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
