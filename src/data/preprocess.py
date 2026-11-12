"""Clean the extracted USCRN hourly data.

Steps (each step's effect is counted in the preprocessing report):
  1. Sort chronologically by UTC timestamp, drop duplicate timestamps.
  2. Convert documented missing-value sentinels (README Note C) to NaN.
  3. Apply documented QC flags (README Note G: 3 = erroneous) -> NaN.
     SUR_TEMP with SUR_TEMP_TYPE == 'U' (unknown/missing, Note H) -> NaN.
  4. Apply hard physical-impossibility bounds from config -> NaN.
  5. Reindex to a continuous hourly UTC timeline (verifies continuity).
  6. Linearly interpolate interior gaps of <= N hours (config), per variable.
     Longer gaps are left missing; precipitation is never interpolated.
  7. Drop rows with any remaining missing selected feature; label contiguous
     hourly runs with `segment_id` so windows never span a gap.
  8. Outlier analysis (IQR / z-score) is reported only - no deletion.

Outputs (data/processed, results/reports, results/graphs).

Usage:
    python -m src.data.preprocess
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.inspect import schema_frame
from src.utils import get_logger, load_config, project_path

log = get_logger("preprocess", "preprocess.log")

FEATURE_LABELS = {
    "T_HR_AVG": "Air temperature (degC)",
    "P_CALC": "Precipitation (mm)",
    "SOLARAD": "Solar radiation (W/m2)",
    "SUR_TEMP": "Surface temperature (degC)",
    "RH_HR_AVG": "Relative humidity (%)",
    "SOIL_TEMP_20": "Soil temperature 20 cm (degC)",
    "SOIL_MOISTURE_5": "Soil moisture 5 cm (m3/m3)",
    "SOIL_MOISTURE_10": "Soil moisture 10 cm (m3/m3)",
    "SOIL_MOISTURE_20": "Soil moisture 20 cm (m3/m3)",
    "SOIL_MOISTURE_50": "Soil moisture 50 cm (m3/m3)",
    "SOIL_MOISTURE_100": "Soil moisture 100 cm (m3/m3)",
}


def nan_runs(mask: np.ndarray) -> np.ndarray:
    """For a boolean NaN mask return, per position, the length of its NaN run (0 if not NaN)."""
    out = np.zeros(len(mask), dtype=int)
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def interpolate_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, int]:
    """Linearly interpolate only interior NaN runs of length <= max_gap."""
    mask = s.isna().to_numpy()
    runs = nan_runs(mask)
    interp = s.interpolate(method="time", limit_area="inside")
    fill = mask & (runs <= max_gap) & interp.notna().to_numpy()
    out = s.copy()
    out[fill] = interp[fill]
    return out, int(fill.sum())


def contiguous_segments(index: pd.DatetimeIndex) -> np.ndarray:
    """Assign an id to each run of consecutive hourly timestamps."""
    # Compare as Timedeltas (resolution-agnostic: pandas may store s/us/ns).
    breaks = np.r_[True, (index[1:] - index[:-1]) != pd.Timedelta(hours=1)]
    return np.cumsum(breaks) - 1


def main() -> int:
    cfg = load_config()
    d, p = cfg["data"], cfg["preprocessing"]
    features: list[str] = cfg["features"]
    target: str = cfg["target"]
    ext = project_path(d["extracted_dir"]) / f"uscrn_hourly_{d['station']}_{d['start_year']}_{d['end_year']}.parquet"
    proc_dir = project_path(d["processed_dir"])
    rep_dir = project_path(cfg["training"]["results_dir"], "reports")
    graph_dir = project_path(cfg["training"]["results_dir"], "graphs")
    for x in (proc_dir, rep_dir, graph_dir):
        x.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ext)
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        log.error("Requested features not in NOAA data: %s", missing_cols)
        return 1

    report: dict = {"input_file": str(ext.name), "n_input_rows": int(len(df)), "steps": {}}

    # 1. sort / dedupe
    df = df.sort_values("UTC_DATETIME")
    n_dup = int(df["UTC_DATETIME"].duplicated().sum())
    df = df.drop_duplicates("UTC_DATETIME", keep="first").set_index("UTC_DATETIME")
    report["steps"]["duplicates_removed"] = n_dup

    keep = features + ["LST_DATETIME", "SUR_TEMP_TYPE"] + list(p["qc_flag_map"].values())
    work = df[keep].copy()

    # 2. sentinels
    schema = schema_frame().set_index("column")
    sent_counts = {}
    for c in features:
        sent = schema.loc[c, "missing_sentinel"]
        m = np.isclose(work[c].astype(float), sent)
        sent_counts[c] = int(m.sum())
        work.loc[m, c] = np.nan
    report["steps"]["sentinel_to_nan"] = sent_counts

    # 3. QC flags
    qc_counts = {}
    for var, flag in p["qc_flag_map"].items():
        if var not in features:
            continue
        m = (work[flag] == "3") & work[var].notna()
        qc_counts[var] = int(m.sum())
        work.loc[m, var] = np.nan
    if "SUR_TEMP" in features:
        m = (work["SUR_TEMP_TYPE"] == "U") & work["SUR_TEMP"].notna()
        qc_counts["SUR_TEMP_type_U"] = int(m.sum())
        work.loc[m, "SUR_TEMP"] = np.nan
    report["steps"]["qc_flag_3_to_nan"] = qc_counts

    # 4. physical bounds
    bound_counts = {}
    for c, (lo, hi) in p["physical_bounds"].items():
        if c not in features:
            continue
        m = work[c].notna() & ((work[c] < lo) | (work[c] > hi))
        bound_counts[c] = int(m.sum())
        work.loc[m, c] = np.nan
    report["steps"]["physical_bounds_to_nan"] = bound_counts

    # 5. continuity
    full_idx = pd.date_range(work.index.min(), work.index.max(), freq="h", name="UTC_DATETIME")
    n_inserted = len(full_idx) - len(work)
    work = work.reindex(full_idx)
    report["steps"]["hourly_timestamps_inserted"] = int(n_inserted)
    missing_before_interp = (work[features].isna().mean() * 100).round(3)

    # 6. short-gap interpolation
    interp_counts = {}
    for c in features:
        if c in p.get("no_interpolation", []):
            interp_counts[c] = 0
            continue
        work[c], interp_counts[c] = interpolate_short_gaps(work[c], p["max_interpolation_gap_hours"])
    report["steps"]["short_gap_values_interpolated"] = interp_counts
    missing_after_interp = (work[features].isna().mean() * 100).round(3)

    # Year-by-feature availability (after cleaning, before row removal)
    avail_year = work[features].notna().groupby(work.index.year).mean().mul(100).round(1)
    avail_year.to_csv(rep_dir / "feature_availability_by_year.csv")

    # 7. drop incomplete rows
    complete = work[features].notna().all(axis=1)
    clean = work.loc[complete, features + ["LST_DATETIME"]].copy()
    clean["segment_id"] = contiguous_segments(clean.index)
    seg_len = clean.groupby("segment_id").size()
    report["steps"]["rows_dropped_incomplete"] = int((~complete).sum())
    report["n_clean_rows"] = int(len(clean))
    report["clean_utc_start"] = str(clean.index.min())
    report["clean_utc_end"] = str(clean.index.max())
    report["clean_rows_by_year"] = {int(k): int(v) for k, v in clean.groupby(clean.index.year).size().items()}
    report["n_segments"] = int(seg_len.size)
    report["segment_length_hours"] = {
        "min": int(seg_len.min()), "median": float(seg_len.median()), "max": int(seg_len.max()),
        "n_segments_ge_25h": int((seg_len >= 25).sum()),
        "rows_in_segments_ge_25h": int(seg_len[seg_len >= 25].sum()),
    }

    # Missing-value report
    mv = pd.DataFrame({
        "sentinel_count": pd.Series(sent_counts),
        "qc_flag_removed": pd.Series({k: v for k, v in qc_counts.items() if k in features}),
        "physical_bound_removed": pd.Series(bound_counts),
        "missing_pct_before_interpolation": missing_before_interp,
        "values_interpolated": pd.Series(interp_counts),
        "missing_pct_after_interpolation": missing_after_interp,
    }).reindex(features).fillna(0)
    mv.to_csv(rep_dir / "missing_value_report.csv")

    # 8. outlier analysis (report only)
    outl = {}
    for c in features:
        s = clean[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        z = (s - s.mean()) / s.std()
        outl[c] = {
            "iqr_outliers_1.5": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            "abs_z_gt_4": int((z.abs() > 4).sum()),
            "min": float(s.min()), "max": float(s.max()),
            "min_time": str(s.idxmin()), "max_time": str(s.idxmax()),
        }
    pd.DataFrame(outl).T.to_csv(rep_dir / "outlier_analysis.csv")
    report["outlier_policy"] = ("Statistical outliers are reported, not removed: extreme weather "
                                "(e.g. heavy rain, cold waves) is physically valid and relevant.")

    stats = clean[features].describe().T
    stats["skew"] = clean[features].skew()
    stats.to_csv(rep_dir / "dataset_statistics.csv")
    corr = clean[features].corr()
    corr.to_csv(rep_dir / "feature_correlation_matrix.csv")

    clean.to_parquet(proc_dir / "cleaned_hourly.parquet")
    clean.to_csv(proc_dir / "cleaned_hourly.csv")
    with open(rep_dir / "preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- Graphs ----
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature correlation (cleaned hourly data)")
    fig.tight_layout(); fig.savefig(graph_dir / "correlation_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(avail_year.T, annot=True, fmt=".0f", cmap="Greens", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "% hours available"})
    ax.set_title("Feature availability by UTC year after QC (before incomplete-row removal)")
    ax.set_xlabel("Year")
    fig.tight_layout(); fig.savefig(graph_dir / "feature_availability_by_year.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(work.index, work[target], lw=0.5, color="#888", label="after QC (all hours)")
    ax.scatter(clean.index, clean[target], s=0.3, color="#1f77b4", label="usable rows (all features present)")
    ax.set_ylabel("VWC (m3/m3)"); ax.set_title(f"{target} over time")
    ax.legend(markerscale=10, loc="lower left")
    fig.tight_layout(); fig.savefig(graph_dir / "target_timeseries.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for ax, c in zip(axes.flat, features):
        sns.histplot(clean[c], bins=60, ax=ax, color="#2a9d8f")
        ax.set_title(FEATURE_LABELS.get(c, c), fontsize=9); ax.set_xlabel("")
    axes.flat[-1].axis("off")
    fig.suptitle("Feature distributions (cleaned)"); fig.tight_layout()
    fig.savefig(graph_dir / "feature_distributions.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    z = (clean[features] - clean[features].mean()) / clean[features].std()
    sns.boxplot(data=z, ax=ax, fliersize=1)
    ax.set_title("Standardised feature boxplots (outliers retained)"); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(graph_dir / "outlier_boxplots.png", dpi=130); plt.close(fig)

    sm_cols = [c for c in features if c.startswith("SOIL_MOISTURE")]
    fig, ax = plt.subplots(figsize=(13, 4))
    for c in sm_cols:
        ax.plot(clean.index, clean[c], ".", ms=0.4, label=c)
    ax.set_title("Soil moisture by depth (usable rows)"); ax.set_ylabel("VWC (m3/m3)")
    ax.legend(markerscale=20, ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(graph_dir / "soil_moisture_profiles.png", dpi=130); plt.close(fig)

    # ---- Console ----
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print("\nMissing-value report:\n", mv.to_string())
    print("\nAvailability by year (%):\n", avail_year.to_string())
    print("=" * 70)
    if len(clean) == 0:
        log.error("No complete rows remain.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
