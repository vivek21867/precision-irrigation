# Deep Learning-Based Precision Irrigation Prediction Framework

**Final-year B.Tech CSE-AIML research project** — *Development of Deep Learning-Based Precision Irrigation Prediction Framework Under Dynamic Environmental Conditions.*

The framework forecasts **next-hour soil moisture at 20 cm** from 24 hours of real meteorological and soil observations from the **NOAA U.S. Climate Reference Network (USCRN)**, station **NE_Lincoln_8_ENE** (Lincoln, Nebraska). It compares **LSTM**, **BiLSTM** and **Transformer-encoder** models under one protocol, and turns the forecast into a configurable **threshold-based irrigation recommendation (YES/NO)**. A Streamlit app presents the data, models, results and explanations.

> Real data only. No synthetic substitution. Every figure in the report is produced by the pipeline.

---

## Results summary

<!-- RESULTS-START -->
Held-out chronological test set: 3,100 next-hour targets, 2021-06-11 → 2021-10-24 (units m³/m³). Full, auto-generated numbers: [`results/reports/EXPERIMENT_REPORT.md`](results/reports/EXPERIMENT_REPORT.md).

| Model | MAE | RMSE | R² | RMSE skill vs persistence |
|---|---|---|---|---|
| Persistence (baseline) | 0.00438 | 0.00835 | 0.9779 | — |
| LSTM | 0.00463 | 0.00732 | 0.9831 | +12.4 % |
| BiLSTM | 0.00428 | 0.00771 | 0.9812 | +7.7 % |
| **Transformer** | **0.00420** | **0.00719** | **0.9837** | **+13.9 %** |

- All three networks (residual formulation) beat persistence on RMSE; the Transformer is best on every metric. LSTM's MAE is slightly worse than persistence: its gains come from large, rain-driven changes.
- **Ablation:** when networks predict the level directly, all three are *worse* than persistence (skill −34 % to −49 %).
- **Irrigation proxy** (threshold 0.364 m³/m³ = training 25th percentile): accuracy ≈ 0.98 and IRRIGATE recall ≈ 0.99 for every model. Because the test period is dry, 93.6 % of hours are IRRIGATE, so these numbers mostly reflect class imbalance. At a balanced threshold (0.273, 5th percentile) the Transformer raises IRRIGATE recall from 0.963 (persistence) to 0.971.
- **Explainability:** after the SM20 history, **precipitation** is the most important input for all models, followed by shallow soil moisture (5/10 cm).
<!-- RESULTS-END -->

---

## Quick start

```bash
cd precision-irrigation
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # CPU-only PyTorch is enough
```

Run the whole pipeline (download → preprocess → windows → train → evaluate → explain → ablation → report):

```bash
python run_pipeline.py
```

Launch the app:

```bash
streamlit run app/main.py
```

Run the tests:

```bash
python -m pytest -q
```

### Running stages individually

| Stage | Command | Main outputs |
|---|---|---|
| Download | `python -m src.data.download` | `data/raw/*.txt`, `data/metadata/download_manifest.csv` |
| Parse & inspect | `python -m src.data.inspect` | `data/extracted/*.parquet/.csv`, `data/metadata/raw_inspection_report.json` |
| Preprocess | `python -m src.data.preprocess` | `data/processed/cleaned_hourly.parquet`, `results/reports/*`, `results/graphs/*` |
| Windows & splits | `python -m src.data.create_sequences` | `data/processed/windows_{train,val,test}.npz`, `scalers.pkl`, `sequence_metadata.json` |
| Train | `python -m src.training.train --model all` | `models/*.pt`, `models/*_config.json`, `results/metrics/*_history.csv` |
| Evaluate | `python -m src.training.evaluate` | `results/metrics/*`, `results/graphs/*` |
| Explain | `python -m src.explainability.feature_importance` | permutation / SHAP CSVs and graphs |
| Report | `python -m src.reporting` | `results/reports/EXPERIMENT_REPORT.md` |

Ablation (direct level prediction): prefix the sequences/train/evaluate commands with `PI_CONFIG=configs/ablation_level_target.yaml` (PowerShell: `$env:PI_CONFIG="configs/ablation_level_target.yaml"`). Outputs go to `models/ablation_level_target/` and `results/ablation_level_target/`.

---

## Data

| | |
|---|---|
| Source | NOAA NCEI USCRN hourly02 — https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/ |
| Documentation | `README.txt` in that directory (saved to `data/metadata/README_hourly02.txt`) |
| Station | NE_Lincoln_8_ENE, WBAN 94995, 40.85° N 96.57° W |
| Files | `CRNH0203-YYYY-NE_Lincoln_8_ENE.txt`, 2015–2024 (10 files, 87,672 hourly records) |

**Input features (11):** `T_HR_AVG` air temperature · `P_CALC` precipitation · `SOLARAD` solar radiation · `SUR_TEMP` surface temperature · `RH_HR_AVG` relative humidity · `SOIL_TEMP_20` · `SOIL_MOISTURE_5/10/20/50/100`.
**Target:** `SOIL_MOISTURE_20` at t+1 h.

**Important limitation.** The soil-moisture sensors have long outages (SM20 missing 2017-03→2019-07; SM5/10/100 absent in 2023–2024). Requiring all 11 features leaves **~21.9 k usable hours (2015–2016, 2019-07→2021-10)**. Details: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §3.

**Cite:** NOAA NCEI, U.S. Climate Reference Network hourly02 product; Diamond et al. (2013) *BAMS* 94:485–498; Bell et al. (2013) *J. Hydrometeor.* 14:977–988.

---

## Method in brief

1. Documented NOAA sentinels → NaN; QC flag 3 → NaN; physical-bounds check; outliers analysed but kept.
2. Linear interpolation of interior gaps ≤ 6 h (never precipitation); remaining incomplete rows dropped; contiguous segments labelled.
3. Chronological 70/15/15 split of the timeline **before** windowing; 24 h → 1 h windows built within each split and segment; scalers fit on training data only.
4. Networks predict the **1-h change** in SM20 (residual formulation), selected on validation because direct-level LSTM/BiLSTM were worse than persistence there. The direct-level variant is reported as an ablation.
5. Identical protocol: MSELoss, Adam, early stopping on validation loss, seed 42.
6. Metrics in m³/m³ (MAE, MSE, RMSE, R²) against a **persistence baseline**.
7. Irrigation: YES if predicted VWC < threshold (default: 25th percentile of training SM20; configurable). Evaluated with accuracy, precision, recall, F1 and confusion matrices against threshold-derived proxy labels.
8. Explainability: permutation importance, GradientSHAP, per-prediction occlusion.

Full methodology: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Web dashboard (React)

A minimalist, light-only dashboard in `web/` (React 18, Tailwind CSS, Framer Motion, dnd-kit, Recharts) following Apple's Human Interface Guidelines: warm-neutral palette with terracotta / sage / sand accents (no blue), translucent cards, Inter / SF Pro typography.

- **Drag and drop** any card to rearrange (layout is remembered per browser; the grid button resets it). Keyboard: focus a card's grip handle, then Space and the arrow keys.
- **Selective expand**: Forecast, Soil Moisture, Accuracy, Decision Quality and Drivers expand in place with spring physics to show drill-down detail. Irrigation, Conditions, Rainfall and Dataset are static summaries. A link like `/#performance` opens a card already expanded.
- **Model switch** in the header recomputes every card for LSTM, BiLSTM or Transformer.
- **Inputs & Output** card: the 11 input features (units, latest value, training range) × 24 hourly steps → next-hour soil moisture at 20 cm (m³/m³) → Irrigate / Hold.
- **Predict page** (`/predict`): enter your own values and get the trained models' forecast live. Three ways to build the 24-hour window the models need:
  - *Latest hour* — your 11 values replace the most recent hour; the previous 23 hours are the observed record.
  - *Steady* — your values are used for all 24 hours (no recent trend).
  - *Full 24 h* — edit every hour; paste rows from a spreadsheet.
  Values outside the physical limits in `config.yaml` are rejected; values outside the training range are accepted with an "extrapolating" warning.

All values come from `web/public/data/dashboard.json`, written by `python -m src.export_dashboard` from the processed NOAA data, trained weights and saved metrics. `run_pipeline.py` regenerates it; nothing in the web app is hard-coded.

The Predict page calls a small local API that runs the PyTorch models (standard library only, localhost):

```bash
python -m src.api                   # http://127.0.0.1:8000  (GET /api/meta, POST /api/predict)
python -m src.export_dashboard      # refresh dashboard data after retraining
cd web
npm install
npm run dev                         # http://localhost:5173
npm run build                       # static site in web/dist/
```

The Streamlit app below remains the analysis workbench (retraining, per-hour predictions, settings).

---

## Streamlit application

| Page | Content |
|---|---|
| Dashboard | Latest observed SM20, next-hour forecast, YES/NO recommendation, current conditions, 7-day history |
| Dataset Explorer | Preview, date ranges, features, missing-value and QC statistics, availability by year, correlations, download manifest, splits |
| Model Training | Saved-model status, loss curves, retrain any model with live progress (optionally re-evaluate) |
| Prediction | Pick model and any held-out hour; prediction vs observed, decision, input window, local explanation |
| Model Comparison | Metrics table and charts, actual vs predicted, error distributions, confusion matrices at the current threshold, threshold sensitivity |
| Explainability | Permutation importance, SHAP by variable and by lag, interpretation |
| Settings | Default model, threshold mode/value (persisted in `app_settings.json`), forecast configuration |

The app reads the real processed data and trained weights; it shows warnings rather than placeholder values if outputs are missing. The "latest" observation is the last complete hour in the processed dataset (2021-10-24), not live data.

---

## Project structure

```
precision-irrigation/
├── config.yaml                  # all settings
├── configs/ablation_level_target.yaml
├── run_pipeline.py              # end-to-end runner
├── data/{raw,extracted,processed,metadata}/
├── src/
│   ├── utils.py                 # config, logging, seeds, paths
│   ├── reporting.py             # builds EXPERIMENT_REPORT.md from outputs
│   ├── export_dashboard.py      # writes web/public/data/dashboard.json
│   ├── api.py                   # local prediction API for the Predict page
│   ├── data/{download,inspect,preprocess,create_sequences}.py
│   ├── models/{lstm,bilstm,transformer}.py
│   ├── training/{train,evaluate}.py
│   ├── prediction/{irrigation,manual}.py
│   └── explainability/feature_importance.py
├── app/{main.py,common.py,pages/}
├── web/                         # React dashboard (src/, public/data/dashboard.json)
├── models/                      # weights + per-model config JSON
├── results/{graphs,metrics,reports}/
├── docs/METHODOLOGY.md
├── tests/test_pipeline.py
├── logs/
└── requirements.txt
```

---

## Limitations

- Single station and soil; ~2.5 years of usable data; test covers Jun–Oct 2021 only.
- 1-hour horizon: persistence is a strong baseline, so gains are reported as skill relative to it.
- Irrigation decisions are a **threshold-based proxy**. USCRN contains **no irrigation actuation labels**; the threshold must be **agronomically calibrated** for a given crop and soil; **no water quantity** is predicted.
- One training seed per model.
- Feature attributions describe model behaviour, not causality.
