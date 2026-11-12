# Research Methodology

**Project:** Development of Deep Learning-Based Precision Irrigation Prediction Framework Under Dynamic Environmental Conditions
**Task:** Forecast next-hour volumetric soil water content (VWC) at 20 cm from the previous 24 h of meteorological and soil observations, then derive a threshold-based irrigation recommendation. Compare LSTM, BiLSTM and Transformer-encoder models under an identical protocol.

All numerical results are in [`results/reports/EXPERIMENT_REPORT.md`](../results/reports/EXPERIMENT_REPORT.md), which is generated from pipeline outputs (`python -m src.reporting`).

---

## 1. Data source and acquisition

| Item | Value |
|---|---|
| Dataset | NOAA NCEI U.S. Climate Reference Network (USCRN), hourly02 product |
| Station | NE_Lincoln_8_ENE, Nebraska (WBAN 94995, 40.85° N, −96.57° E) |
| Period requested | 2015–2024 (10 annual files, UTC years) |
| Archive | https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/ |
| Documentation | https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/README.txt (copy in `data/metadata/`) |

`src/data/download.py` reads the official directory listing for each year and selects the station file by the documented naming convention `CRNH02TT-YYYY-${name}.txt` (no guessed URLs). Downloads use retries with back-off and HTTP Range resume; each file's size is checked against the server's `Content-Length`, every line is checked against the documented record layout, and a SHA-256 hash is recorded. `data/metadata/download_manifest.csv` lists filename, URL, year, sizes, hash, line counts and status. Raw files in `data/raw/` are never modified.

**Citation.** Diamond, H. J., et al. (2013). U.S. Climate Reference Network after one decade of operations: status and assessment. *Bull. Amer. Meteor. Soc.*, 94, 485–498. doi:10.1175/BAMS-D-12-00170.1. Bell, J. E., et al. (2013). U.S. Climate Reference Network soil moisture and temperature observations. *J. Hydrometeorology*, 14, 977–988. doi:10.1175/JHM-D-12-0146.1. Data: NOAA NCEI, USCRN hourly02 product, accessed 2026-10-31.

## 2. Parsing and quality control

Fields are parsed with the fixed column positions documented in README §5 (`src/data/inspect.py`). Only documented rules are applied:

1. **Missing-value sentinels** (README Note C — "lowest possible integer for a given column format"): derived from each field's width/decimals, e.g. −9999.0 (7 chars, 1 dp), −99.000 (7 chars, 3 dp), −99999 (6-char integer). Converted to NaN.
2. **QC flags** (Note G — 0 = good, 3 = erroneous): values of SOLARAD, SUR_TEMP and RH_HR_AVG with flag 3 are set to NaN. Flagged values were confirmed to hold plausible-looking numbers (e.g. SOLARAD = 561 W/m² with flag 3), so applying flags matters. Calculated fields T_HR_AVG and P_CALC carry no flags (Note E: reported as missing when raw data are bad). Soil variables have no flag in this product.
3. **SUR_TEMP_TYPE = 'U'** (Note H — unknown/missing) → SUR_TEMP NaN (in practice these rows were already sentinels).
4. **Physical-impossibility bounds** (e.g. VWC ∉ [0, 1], RH ∉ [0, 100], negative precipitation) → NaN. None were triggered.

**Outliers** are analysed (IQR and |z| > 4 counts in `results/reports/outlier_analysis.csv`) but **not removed**: the −36.3 °C air temperature corresponds to the February 2021 cold wave, and heavy-rain hours drive the wetting events we need to model.

## 3. Gap handling and continuity

- Timeline sorted by UTC; duplicate timestamps removed (none found); reindexed to a continuous hourly grid (no missing timestamps in the raw files).
- **Interpolation:** linear, per variable, only for *interior* gaps of ≤ 6 h. Longer gaps are left missing. **Precipitation is never interpolated** — it is an intermittent accumulation, so interpolating it would fabricate rainfall.
- Rows still missing any selected feature are removed. Remaining rows are labelled with `segment_id` (runs of consecutive hours), and windows are only built inside a segment.

### Key data limitation

The soil-moisture sensors have long outages (`results/graphs/feature_availability_by_year.png`):

| Variable | Not available |
|---|---|
| SOIL_MOISTURE_20 (target) | 2017-03 → 2019-07; 2021-10 → 2022-06 |
| SOIL_MOISTURE_5 / _10 | most of 2023; all of 2024 |
| SOIL_MOISTURE_100 | 2023–2024 |
| SOIL_MOISTURE_50 | most of 2016–2017 |

Requiring all 11 requested features therefore restricts usable data to **2015–2016 and 2019-07 → 2021-10** (≈21.9 k hours of 87.7 k). The 2022–2024 records of SM20 exist but cannot be used with the 11-feature set because SM5/10/100 are absent. In addition, SM20 after the 2021–22 outage sits at a visibly lower level (≈0.20–0.35 vs 0.35–0.50 m³/m³), consistent with a sensor/installation change; this is an inference from the data, not documented in the README. No synthetic data were substituted.

## 4. Supervised dataset

- **Window:** inputs = hours t−23 … t (24 h × 11 features); target = SM20 at t+1; stride 1 h.
- **Split before windowing:** cut timestamps are placed at 70 % and 85 % of the usable, window-eligible hours in time order. Every row is assigned to train/val/test by its timestamp, segments are cut at the boundaries, and windows are built separately in each split. Assertions verify that no window's inputs or target cross a boundary and that stored windows are in strictly increasing time order.
- Because the calendar range 2015–2024 has no usable data after 2021-10, splitting by **usable** hours (rather than calendar time) is the only way to obtain non-empty validation and test sets; the order is still strictly chronological.
- **Scaling:** `StandardScaler` for inputs and the network target, fitted **only on training rows**.
- **Distribution shift:** the test period (2021-06 → 2021-10) is a summer–autumn dry-down with mean SM20 ≈ 0.28 m³/m³ versus ≈ 0.39 in training — a demanding out-of-sample test.

## 5. Target formulation (decided on validation data)

Two formulations were implemented (`sequences.target_mode`):

- **level** — network outputs SM20(t+1) directly;
- **delta (residual)** — network outputs ΔSM20 = SM20(t+1) − SM20(t); the forecast is SM20(t) + Δ̂.

In a first run with the *level* formulation, LSTM and BiLSTM had **higher validation RMSE than naive persistence** (≈0.010–0.011 vs 0.0063 m³/m³). Soil moisture is strongly autocorrelated at a 1-h horizon, and a network regressing the level must reconstruct the current value through its nonlinearity, leaving an error floor. The residual formulation is the standard remedy, so it was adopted as the **primary** experiment. This decision used validation data only; the test set was not consulted. Both formulations are reported (the level variant as an ablation, `configs/ablation_level_target.yaml`).

## 6. Models (PyTorch, CPU)

| Model | Architecture (config.yaml) |
|---|---|
| LSTM | 2-layer LSTM, hidden 64, dropout 0.2 → last hidden state → linear |
| BiLSTM | 2-layer bidirectional LSTM, hidden 64/direction, dropout 0.2 → concat(final fwd, final bwd) → linear. Bidirectionality is over the past 24-h window only. |
| Transformer | Linear input projection (11→64) → sinusoidal positional encoding → 2 pre-norm encoder layers (4 heads, FFN 128, dropout 0.1, LayerNorm) → final LayerNorm → last-step token → linear |

**Identical protocol:** MSELoss on the standardised target, Adam (lr 1e-3), batch 256, gradient-norm clipping 1.0, max 60 epochs, early stopping on validation loss with patience 8 and best-weight restore, seed 42, `torch.use_deterministic_algorithms(True)`. The training loader shuffles the *order of training windows* between mini-batches (standard SGD practice); each window keeps its internal chronological order, and validation/test are never shuffled, so no future data reach training.

## 7. Evaluation

Metrics are computed in physical units (m³/m³) on the held-out test windows: MAE, MSE, RMSE, R². Because R² is inflated for persistent series, a **persistence baseline** (ŷ(t+1) = y(t)) is always reported, along with **RMSE skill = 1 − RMSE_model / RMSE_persistence**.

## 8. Irrigation decision module

`src/prediction/irrigation.py`: IRRIGATE (YES) if predicted VWC < threshold, otherwise NO. The threshold is configurable — either a fixed VWC value or (default) the 25th percentile of the **training-period** SM20 distribution.

**Limitations (stated explicitly):**
- USCRN has **no irrigation-event labels**. Reference labels are generated by applying the same threshold to the *observed* next-hour VWC, so the classification metrics measure how well forecasts reproduce threshold crossings — not agreement with real irrigation practice.
- The default threshold is a statistical proxy. A deployable threshold must be agronomically calibrated (field capacity, wilting point, management-allowed depletion, crop stage, rooting depth).
- The system does **not** estimate irrigation water quantity.
- Because the test period is drier than training, most test hours fall below the training 25th percentile; the class balance is reported, and recall for IRRIGATE is emphasised. A threshold sweep (5th–50th training percentiles) shows sensitivity.

## 9. Explainability

- **Permutation importance** (complete forecaster): each variable's whole 24-h sequence is permuted across test windows; importance = mean increase in test RMSE over 3 repeats.
- **GradientSHAP** (expected gradients, `shap.GradientExplainer`, 100 training windows as background, 200 test windows explained): explains the *network output* — in the residual formulation, the predicted 1-h change. Reported per variable and per (lag, variable).
- **Local occlusion** (app): change in prediction when one variable is replaced by its training mean.

These are associations learned by the models, not causal effects.

## 10. Reproducibility

Python 3.11, PyTorch 2.x CPU. `python run_pipeline.py` reproduces everything from download to report. Seeds, configs, scalers, weights, per-epoch histories and logs are saved (`models/`, `data/processed/`, `results/`, `logs/`).

## 11. Threats to validity

1. Single station, single soil; results may not transfer.
2. Usable data restricted to ~2.5 years by sensor outages; the test set covers one season (Jun–Oct 2021).
3. One hour is a very short horizon for irrigation management; persistence is a strong baseline. Multi-hour/day horizons are the natural extension (`sequences.horizon`).
4. Single training seed per model; differences between models smaller than seed-to-seed variance should not be over-interpreted.
5. Irrigation labels are proxies (Section 8).
