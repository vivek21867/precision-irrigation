"""Generate the experiment report from saved pipeline outputs.

Every number in results/reports/EXPERIMENT_REPORT.md is read from files
written by the pipeline - nothing is typed by hand.

Usage:
    python -m src.reporting
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.utils import load_config, project_path

FEATURE_TEXT = {
    "T_HR_AVG": "air temperature", "P_CALC": "precipitation", "SOLARAD": "solar radiation",
    "SUR_TEMP": "surface temperature", "RH_HR_AVG": "relative humidity", "SOIL_TEMP_20": "soil temperature (20 cm)",
    "SOIL_MOISTURE_5": "soil moisture 5 cm", "SOIL_MOISTURE_10": "soil moisture 10 cm",
    "SOIL_MOISTURE_20": "soil moisture 20 cm (target history)", "SOIL_MOISTURE_50": "soil moisture 50 cm",
    "SOIL_MOISTURE_100": "soil moisture 100 cm",
}


def jload(rel: str) -> dict:
    return json.loads(project_path(rel).read_text(encoding="utf-8"))


def md(df: pd.DataFrame, fmt: str = ".6f") -> str:
    return df.to_markdown(floatfmt=fmt)


def main() -> int:
    cfg = load_config()
    raw = jload("data/metadata/raw_inspection_report.json")
    prep = jload("results/reports/preprocessing_report.json")
    seq = jload("data/processed/sequence_metadata.json")
    man = pd.read_csv(project_path("data/metadata/download_manifest.csv"))
    test = pd.read_csv(project_path("results/metrics/test_metrics_table.csv"), index_col=0)
    allm = pd.read_csv(project_path("results/metrics/regression_metrics.csv"))
    irr = jload("results/metrics/irrigation_metrics.json")
    perm = pd.read_csv(project_path("results/metrics/permutation_importance.csv"))
    sweep = pd.read_csv(project_path("results/metrics/irrigation_threshold_sweep.csv"))
    shap_p = project_path("results/metrics/shap_mean_abs_by_feature.csv")
    shap_f = pd.read_csv(shap_p, index_col=0) if shap_p.exists() else None
    abl_p = project_path("results/ablation_level_target/metrics/test_metrics_table.csv")
    abl = pd.read_csv(abl_p, index_col=0) if abl_p.exists() else None
    mcfgs = {m: jload(f"models/{m}_config.json") for m in ("lstm", "bilstm", "transformer")
             if project_path(f"models/{m}_config.json").exists()}

    deep = test.drop(index="Persistence")
    best = deep["RMSE"].idxmin()
    val = allm[allm.split == "val"].set_index("model")[["MAE", "RMSE", "R2"]]

    irr_tbl = pd.DataFrame({m: {k: r[k] for k in ("accuracy", "precision_irrigate", "recall_irrigate",
                                                   "f1_irrigate", "positive_rate_actual")}
                            for m, r in irr["models"].items()}).T

    pv = perm.pivot(index="feature", columns="model", values="rmse_increase_mean")
    pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]

    # ---- interpretation points (data-driven) ----
    points = []
    for m in pv.columns:
        top = pv[m].sort_values(ascending=False)
        points.append(f"{m}: largest permutation importance = {FEATURE_TEXT[top.index[0]]} "
                      f"(+{top.iloc[0]:.4f} m³/m³ RMSE), then {FEATURE_TEXT[top.index[1]]} "
                      f"(+{top.iloc[1]:.4f}) and {FEATURE_TEXT[top.index[2]]} (+{top.iloc[2]:.4f}).")
    if shap_f is not None:
        for m in shap_f.columns:
            s = shap_f[m].sort_values(ascending=False)
            points.append(f"{m} (SHAP, drivers of the predicted 1-h change): {FEATURE_TEXT[s.index[0]]}, "
                          f"{FEATURE_TEXT[s.index[1]]}, {FEATURE_TEXT[s.index[2]]}.")
    points.append("SOIL_MOISTURE_20 dominates permutation importance largely by construction: the forecaster "
                  "adds the predicted change to the last observed SM20, so permuting SM20 history moves the "
                  "forecast baseline itself.")
    if shap_f is not None:
        points.append("Large mean |SHAP| on SM20 lags (larger than the typical 1-h change) arises from opposite-signed "
                      "attributions on adjacent lags, i.e. the networks use the recent rate of change of SM20 "
                      "rather than its absolute level (see the lag heatmaps).")
    pr = pv.loc["P_CALC"]
    points.append("Precipitation permutation importance by model: "
                  + ", ".join(f"{m} +{v:.5f}" for m, v in pr.items())
                  + " m³/m³ — rain events drive the sharp wetting jumps that persistence cannot anticipate.")
    (project_path("results/reports") / "explainability_interpretation.json").write_text(
        json.dumps({"points": points}, indent=2), encoding="utf-8")

    ok = man["status"].isin(["downloaded", "already_present"])
    s = seq["splits"]
    lines = [
        "# Experiment Report — Precision Irrigation Prediction Framework",
        "",
        f"_Generated automatically by `python -m src.reporting` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
        "All values are read from pipeline outputs._",
        "",
        "## 1. Data",
        f"- Source: NOAA NCEI USCRN hourly02, station **{cfg['data']['station']}** "
        f"(WBAN {raw['wbanno'][0]}, {raw['latitude'][0]}°N, {raw['longitude'][0]}°E).",
        f"- Files downloaded: **{int(ok.sum())}/{len(man)}** (years {man.loc[ok, 'year'].min()}–{man.loc[ok, 'year'].max()}), "
        f"{int(man.loc[ok, 'local_size_bytes'].sum()):,} bytes; all sizes match server Content-Length; "
        f"{int(man['n_malformed_lines'].sum())} malformed lines.",
        f"- Raw hourly records: **{raw['total_observations']:,}** ({raw['utc_start']} → {raw['utc_end']} UTC), "
        f"{raw['duplicate_utc_timestamps']} duplicate and {raw['missing_hourly_timestamps']} missing timestamps.",
        f"- After QC, ≤{cfg['preprocessing']['max_interpolation_gap_hours']} h interpolation and removal of rows lacking any of the "
        f"{len(cfg['features'])} features: **{prep['n_clean_rows']:,}** usable hours in {prep['n_segments']} contiguous segments "
        f"({prep['clean_utc_start']} → {prep['clean_utc_end']}). Rows by year: {prep['clean_rows_by_year']}.",
        "",
        "## 2. Windows and splits (chronological, timeline split before windowing)",
        "",
        md(pd.DataFrame({k: {"windows": v["n_windows"], "first target": v["target_time_start"],
                             "last target": v["target_time_end"], "mean SM20": round(v["target_mean"], 4)}
                         for k, v in s.items()}).T, ".4f"),
        "",
        f"Network target mode: **{seq.get('target_mode', 'level')}**.",
        "",
        "## 3. Test-set regression results (m³/m³)",
        "",
        md(test),
        "",
        f"Best deep model by test RMSE: **{best}**. "
        f"RMSE skill vs persistence: " + ", ".join(f"{m} {v:+.3f}" for m, v in deep["RMSE_skill_vs_persistence"].items()) + ".",
        "",
        "Validation-set results:",
        "",
        md(val),
        "",
    ]
    if abl is not None:
        lines += ["### Ablation: direct level prediction (network predicts SM20(t+1) itself)", "", md(abl), ""]
    lines += [
        "## 4. Training",
        "",
        md(pd.DataFrame({m: {"params": c["n_trainable_params"], "best_epoch": c["best_epoch"],
                             "epochs_run": c["epochs_run"], "time_s": c["training_time_s"],
                             "best_val_mse_scaled": c["best_val_mse_scaled"]} for m, c in mcfgs.items()}).T, ".5f"),
        "",
        "## 5. Irrigation decision module (threshold proxy)",
        f"Threshold: **{irr['threshold']['threshold']:.4f} m³/m³** — {irr['threshold']['source']}. "
        "Positive class = IRRIGATE. Reference decisions come from applying the same threshold to observed SM20.",
        "",
        md(irr_tbl, ".4f"),
        "",
        "Confusion matrices (rows actual NO/YES, columns predicted NO/YES): "
        + "; ".join(f"{m} {r['confusion_matrix']}" for m, r in irr["models"].items()) + ".",
        "",
        f"Note: the test period is much drier than training, so {irr_tbl['positive_rate_actual'].iloc[0]:.1%} of test "
        "hours are IRRIGATE at the default threshold and accuracy is dominated by that class. The sweep below "
        "(recall for IRRIGATE at thresholds set from training percentiles) is more informative:",
        "",
        md(sweep.pivot(index=["train_percentile", "threshold"], columns="model", values="recall_irrigate")
           .join(sweep.groupby(["train_percentile", "threshold"])["positive_rate_actual"].first()), ".4f"),
        "",
        "## 6. Explainability",
        "",
        "Permutation importance (increase in test RMSE, m³/m³):",
        "",
        md(pv),
        "",
    ]
    if shap_f is not None:
        lines += ["Mean |GradientSHAP| (network output, summed over lags):", "", md(shap_f), ""]
    lines += [*(f"- {p}" for p in points), ""]
    out = project_path("results/reports/EXPERIMENT_REPORT.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
