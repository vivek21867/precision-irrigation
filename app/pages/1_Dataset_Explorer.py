"""Dataset Explorer - real NOAA USCRN data, provenance and quality statistics."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import FEATURE_LABELS, cleaned, config, page_setup, read_csv, read_json, require

page_setup("Dataset Explorer", "📊")
st.title("📊 Dataset Explorer")
if not require(["data/processed/cleaned_hourly.parquet"], "Run the data pipeline first."):
    st.stop()

cfg = config()
features, target = cfg["features"], cfg["target"]
df = cleaned()
raw = read_json("data/metadata/raw_inspection_report.json") or {}
prep = read_json("results/reports/preprocessing_report.json") or {}
seq = read_json("data/processed/sequence_metadata.json") or {}

st.markdown("**Source:** NOAA NCEI U.S. Climate Reference Network (USCRN) hourly02 product, station "
            f"`{cfg['data']['station']}` (WBAN {', '.join(raw.get('wbanno', []))}, "
            f"lat {raw.get('latitude', ['?'])[0]}, lon {raw.get('longitude', ['?'])[0]}). "
            "[Archive](https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/)")

c = st.columns(5)
c[0].metric("Raw hourly records", f"{raw.get('total_observations', 0):,}")
c[1].metric("Usable rows (all features)", f"{len(df):,}")
c[2].metric("Contiguous segments", prep.get("n_segments", "?"))
c[3].metric("Raw UTC range", f"{raw.get('utc_start', '')[:10]} → {raw.get('utc_end', '')[:10]}")
c[4].metric("Usable UTC range", f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preview", "Time series", "Missing values & QC",
                                        "Statistics & correlation", "Download manifest & splits"])
with tab1:
    st.write("Selected features:", ", ".join(f"`{f}`" for f in features), "→ target:", f"`{target}` (next hour)")
    st.dataframe(df.head(500), use_container_width=True, height=380)
    st.caption("Showing the first 500 cleaned rows. Times are UTC (end of the observed hour).")

with tab2:
    f = st.selectbox("Variable", features, index=features.index(target), format_func=lambda x: FEATURE_LABELS[x])
    rng = st.slider("Date range", df.index.min().to_pydatetime(), df.index.max().to_pydatetime(),
                    (df.index.min().to_pydatetime(), df.index.max().to_pydatetime()))
    sub = df.loc[rng[0]:rng[1]]
    fig = px.scatter(sub, x=sub.index, y=f, labels={"x": "UTC", f: FEATURE_LABELS[f]}, template="plotly_white")
    fig.update_traces(marker=dict(size=2, color="#2a9d8f"))
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points rather than lines: gaps (removed long outages) are left visible, never filled.")

with tab3:
    mv = read_csv("results/reports/missing_value_report.csv", index_col=0)
    if mv is not None:
        st.markdown("**Missing values per stage** (NOAA sentinel → QC flag 3 → physical bounds → ≤6 h interpolation)")
        st.dataframe(mv, use_container_width=True)
    av = read_csv("results/reports/feature_availability_by_year.csv", index_col=0)
    if av is not None:
        fig = px.imshow(av.T, text_auto=".0f", color_continuous_scale="Greens", aspect="auto",
                        labels=dict(color="% available", x="Year"))
        fig.update_layout(height=430, title="Feature availability by year after QC (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.warning("Soil-moisture sensors have long outages: SM 20 cm is missing from 2017-03 to 2019-07; "
                   "SM 5/10 cm are absent from mid-2023 and SM 100 cm from 2023. Requiring all 11 features "
                   "therefore limits usable data to 2015–2016 and 2019–2021.")
    st.json({"QC flags (README Note G: 0 good, 3 erroneous)": raw.get("qc_flag_value_counts", {}),
             "Preprocessing steps": prep.get("steps", {})}, expanded=False)

with tab4:
    stats = read_csv("results/reports/dataset_statistics.csv", index_col=0)
    if stats is not None:
        st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)
    corr = df[features].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
    out = read_csv("results/reports/outlier_analysis.csv", index_col=0)
    if out is not None:
        st.markdown("**Outlier analysis** (reported only — extreme weather is retained)")
        st.dataframe(out, use_container_width=True)

with tab5:
    man = read_csv("data/metadata/download_manifest.csv")
    if man is not None:
        st.dataframe(man[["year", "filename", "remote_size_bytes", "local_size_bytes", "status",
                          "n_lines", "n_malformed_lines", "url"]], use_container_width=True)
    if seq:
        st.markdown("**Chronological splits** (timeline split before windowing; 24 h input → next hour)")
        st.dataframe([{"split": k, **{kk: v[kk] for kk in ("rows", "n_windows", "X_shape",
                                                             "target_time_start", "target_time_end", "target_mean")}}
                      for k, v in seq["splits"].items()], use_container_width=True)
