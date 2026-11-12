"""Parse and inspect the raw USCRN hourly02 files.

Column positions, widths and decimals come from README.txt section 5
(saved in data/metadata/README_hourly02.txt). Missing-value sentinels are
derived from the documented rule (Note C): "the lowest possible integer for a
given column format", e.g. -9999.0 for 7-char/1-decimal fields and -99.000 for
7-char/3-decimal fields.

Outputs:
    data/extracted/uscrn_hourly_<station>_<start>_<end>.parquet  (raw values, typed)
    data/extracted/uscrn_hourly_<station>_<start>_<end>.csv
    data/metadata/raw_inspection_report.json
    data/metadata/column_schema.csv

Usage:
    python -m src.data.inspect
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, project_path

log = get_logger("inspect", "inspect.log")

# (name, start_col, end_col [1-based inclusive, from README], kind, decimals, unit)
# kind: "str" (text), "flag" (QC flag / type char), "num"
SCHEMA = [
    ("WBANNO", 1, 5, "str", None, "XXXXX"),
    ("UTC_DATE", 7, 14, "str", None, "YYYYMMDD"),
    ("UTC_TIME", 16, 19, "str", None, "HHmm"),
    ("LST_DATE", 21, 28, "str", None, "YYYYMMDD"),
    ("LST_TIME", 30, 33, "str", None, "HHmm"),
    ("CRX_VN", 35, 40, "str", None, "XXXXXX"),
    ("LONGITUDE", 42, 48, "num", 2, "Decimal_degrees"),
    ("LATITUDE", 50, 56, "num", 2, "Decimal_degrees"),
    ("T_CALC", 58, 64, "num", 1, "Celsius"),
    ("T_HR_AVG", 66, 72, "num", 1, "Celsius"),
    ("T_MAX", 74, 80, "num", 1, "Celsius"),
    ("T_MIN", 82, 88, "num", 1, "Celsius"),
    ("P_CALC", 90, 96, "num", 1, "mm"),
    ("SOLARAD", 98, 103, "num", 0, "W/m^2"),
    ("SOLARAD_FLAG", 105, 105, "flag", None, "X"),
    ("SOLARAD_MAX", 107, 112, "num", 0, "W/m^2"),
    ("SOLARAD_MAX_FLAG", 114, 114, "flag", None, "X"),
    ("SOLARAD_MIN", 116, 121, "num", 0, "W/m^2"),
    ("SOLARAD_MIN_FLAG", 123, 123, "flag", None, "X"),
    ("SUR_TEMP_TYPE", 125, 125, "flag", None, "X"),
    ("SUR_TEMP", 127, 133, "num", 1, "Celsius"),
    ("SUR_TEMP_FLAG", 135, 135, "flag", None, "X"),
    ("SUR_TEMP_MAX", 137, 143, "num", 1, "Celsius"),
    ("SUR_TEMP_MAX_FLAG", 145, 145, "flag", None, "X"),
    ("SUR_TEMP_MIN", 147, 153, "num", 1, "Celsius"),
    ("SUR_TEMP_MIN_FLAG", 155, 155, "flag", None, "X"),
    ("RH_HR_AVG", 157, 161, "num", 0, "%"),
    ("RH_HR_AVG_FLAG", 163, 163, "flag", None, "X"),
    ("SOIL_MOISTURE_5", 165, 171, "num", 3, "m^3/m^3"),
    ("SOIL_MOISTURE_10", 173, 179, "num", 3, "m^3/m^3"),
    ("SOIL_MOISTURE_20", 181, 187, "num", 3, "m^3/m^3"),
    ("SOIL_MOISTURE_50", 189, 195, "num", 3, "m^3/m^3"),
    ("SOIL_MOISTURE_100", 197, 203, "num", 3, "m^3/m^3"),
    ("SOIL_TEMP_5", 205, 211, "num", 1, "Celsius"),
    ("SOIL_TEMP_10", 213, 219, "num", 1, "Celsius"),
    ("SOIL_TEMP_20", 221, 227, "num", 1, "Celsius"),
    ("SOIL_TEMP_50", 229, 235, "num", 1, "Celsius"),
    ("SOIL_TEMP_100", 237, 243, "num", 1, "Celsius"),
]


def sentinel_for(width: int, decimals: int) -> float:
    """Lowest value representable in a fixed-width field (README Note C)."""
    int_digits = width - 1 - (decimals + 1 if decimals else 0)
    return -(10 ** int_digits - 1) * 1.0


def schema_frame() -> pd.DataFrame:
    rows = []
    for name, s, e, kind, dec, unit in SCHEMA:
        w = e - s + 1
        rows.append({"column": name, "start": s, "end": e, "width": w, "kind": kind,
                     "decimals": dec, "unit": unit,
                     "missing_sentinel": sentinel_for(w, dec) if kind == "num" else None})
    return pd.DataFrame(rows)


def parse_file(path) -> pd.DataFrame:
    colspecs = [(s - 1, e) for _, s, e, *_ in SCHEMA]
    names = [n for n, *_ in SCHEMA]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str, header=None)
    for name, _, _, kind, _, _ in SCHEMA:
        if kind == "num":
            df[name] = pd.to_numeric(df[name], errors="raise")
        else:
            df[name] = df[name].astype("string").str.strip()
    df["UTC_DATETIME"] = pd.to_datetime(df["UTC_DATE"] + df["UTC_TIME"], format="%Y%m%d%H%M")
    df["LST_DATETIME"] = pd.to_datetime(df["LST_DATE"] + df["LST_TIME"], format="%Y%m%d%H%M")
    df["SOURCE_FILE"] = path.name
    return df


def main() -> int:
    cfg = load_config()
    d = cfg["data"]
    raw_dir = project_path(d["raw_dir"])
    ext_dir = project_path(d["extracted_dir"])
    meta_dir = project_path(d["metadata_dir"])
    ext_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(meta_dir / "download_manifest.csv")
    ok = manifest[manifest["status"].isin(["downloaded", "already_present"])]
    if ok.empty:
        log.error("No successfully downloaded files in manifest - run download first.")
        return 1

    frames = [parse_file(raw_dir / f) for f in ok["filename"]]
    df = pd.concat(frames, ignore_index=True)
    schema = schema_frame()
    schema.to_csv(meta_dir / "column_schema.csv", index=False)

    stem = f"uscrn_hourly_{d['station']}_{d['start_year']}_{d['end_year']}"
    df.to_parquet(ext_dir / f"{stem}.parquet", index=False)
    df.to_csv(ext_dir / f"{stem}.csv", index=False)

    # ---- Inspection ----
    num_cols = schema.loc[schema.kind == "num", ["column", "missing_sentinel"]]
    missing = {}
    observed_min = {}
    for col, sent in num_cols.itertuples(index=False):
        is_missing = np.isclose(df[col], sent)
        missing[col] = round(100 * is_missing.mean(), 3)
        valid = df.loc[~is_missing, col]
        observed_min[col] = float(valid.min()) if len(valid) else None
    flag_cols = schema.loc[schema.kind == "flag", "column"]
    flag_counts = {c: df[c].value_counts(dropna=False).to_dict() for c in flag_cols}

    dup = int(df["UTC_DATETIME"].duplicated().sum())
    full_range = pd.date_range(df["UTC_DATETIME"].min(), df["UTC_DATETIME"].max(), freq="h")
    gaps = len(full_range) - df["UTC_DATETIME"].nunique()

    report = {
        "station": d["station"],
        "wbanno": sorted(df["WBANNO"].unique().tolist()),
        "latitude": sorted(df["LATITUDE"].unique().tolist()),
        "longitude": sorted(df["LONGITUDE"].unique().tolist()),
        "n_files": int(len(ok)),
        "years_downloaded": ok["year"].astype(int).tolist(),
        "total_observations": int(len(df)),
        "utc_start": str(df["UTC_DATETIME"].min()),
        "utc_end": str(df["UTC_DATETIME"].max()),
        "duplicate_utc_timestamps": dup,
        "missing_hourly_timestamps": int(gaps),
        "columns": list(df.columns),
        "missing_percent_by_column": missing,
        "observed_min_excluding_sentinel": observed_min,
        "qc_flag_value_counts": {k: {str(kk): int(vv) for kk, vv in v.items()} for k, v in flag_counts.items()},
        "datalogger_versions": df["CRX_VN"].value_counts().to_dict(),
        "soil_moisture_columns": [c for c in df.columns if c.startswith("SOIL_MOISTURE")],
    }
    with open(meta_dir / "raw_inspection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ---- Console summary ----
    print("=" * 70)
    print(f"Station: {d['station']}  WBAN={report['wbanno']}  lat={report['latitude']} lon={report['longitude']}")
    print(f"Downloaded files: {report['n_files']}   Years: {report['years_downloaded']}")
    print(f"Total observations: {report['total_observations']}")
    print(f"UTC date range: {report['utc_start']} -> {report['utc_end']}")
    print(f"Duplicate timestamps: {dup}   Missing hourly timestamps: {gaps}")
    print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")
    print("\nMissing-value % (documented sentinels):")
    for c, p in missing.items():
        print(f"  {c:20s} {p:7.3f}%")
    print("\nQC flags (README Note G: 0=good, 3=erroneous):")
    for c, v in report["qc_flag_value_counts"].items():
        print(f"  {c:20s} {v}")
    print("\nSoil moisture measurements:", report["soil_moisture_columns"])
    print("=" * 70)
    log.info("Extracted dataset written to %s", ext_dir / f"{stem}.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
