# Experiment Report — Precision Irrigation Prediction Framework

_Generated automatically by `python -m src.reporting` on 2026-10-31 23:11. All values are read from pipeline outputs._

## 1. Data
- Source: NOAA NCEI USCRN hourly02, station **NE_Lincoln_8_ENE** (WBAN 94995, 40.85°N, -96.57°E).
- Files downloaded: **10/10** (years 2015–2024), 21,391,968 bytes; all sizes match server Content-Length; 0 malformed lines.
- Raw hourly records: **87,672** (2015-01-01 01:00:00 → 2025-01-01 00:00:00 UTC), 0 duplicate and 0 missing timestamps.
- After QC, ≤6 h interpolation and removal of rows lacking any of the 11 features: **21,932** usable hours in 86 contiguous segments (2015-05-01 01:00:00 → 2021-10-24 18:00:00). Rows by year: {'2015': 4796, '2016': 370, '2019': 3523, '2020': 7078, '2021': 6165}.

## 2. Windows and splits (chronological, timeline split before windowing)

|       |   windows | first target        | last target         |   mean SM20 |
|:------|----------:|:--------------------|:--------------------|------------:|
| train |     14174 | 2015-05-02 01:00:00 | 2020-12-08 06:00:00 |      0.3914 |
| val   |      3047 | 2020-12-09 19:00:00 | 2021-06-10 20:00:00 |      0.3858 |
| test  |      3100 | 2021-06-11 21:00:00 | 2021-10-24 18:00:00 |      0.2786 |

Network target mode: **delta**.

## 3. Test-set regression results (m³/m³)

| model       |      MAE |      MSE |     RMSE |       R2 |   RMSE_skill_vs_persistence |
|:------------|---------:|---------:|---------:|---------:|----------------------------:|
| Persistence | 0.004378 | 0.000070 | 0.008351 | 0.977944 |                    0.000000 |
| LSTM        | 0.004633 | 0.000054 | 0.007318 | 0.983065 |                    0.123727 |
| BiLSTM      | 0.004283 | 0.000059 | 0.007706 | 0.981222 |                    0.077283 |
| Transformer | 0.004196 | 0.000052 | 0.007188 | 0.983660 |                    0.139270 |

Best deep model by test RMSE: **Transformer**. RMSE skill vs persistence: LSTM +0.124, BiLSTM +0.077, Transformer +0.139.

Validation-set results:

| model       |      MAE |     RMSE |       R2 |
|:------------|---------:|---------:|---------:|
| Persistence | 0.004336 | 0.006329 | 0.990411 |
| LSTM        | 0.004407 | 0.006152 | 0.990939 |
| BiLSTM      | 0.004306 | 0.006100 | 0.991092 |
| Transformer | 0.004134 | 0.005829 | 0.991867 |

### Ablation: direct level prediction (network predicts SM20(t+1) itself)

| model       |      MAE |      MSE |     RMSE |       R2 |   RMSE_skill_vs_persistence |
|:------------|---------:|---------:|---------:|---------:|----------------------------:|
| Persistence | 0.004378 | 0.000070 | 0.008351 | 0.977944 |                    0.000000 |
| LSTM        | 0.008515 | 0.000125 | 0.011179 | 0.960483 |                   -0.338553 |
| BiLSTM      | 0.009208 | 0.000155 | 0.012432 | 0.951127 |                   -0.488598 |
| Transformer | 0.008074 | 0.000130 | 0.011392 | 0.958959 |                   -0.364111 |

## 4. Training

|             |       params |   best_epoch |   epochs_run |    time_s |   best_val_mse_scaled |
|:------------|-------------:|-------------:|-------------:|----------:|----------------------:|
| lstm        |  53057.00000 |     10.00000 |     18.00000 |  58.40000 |               0.44604 |
| bilstm      | 138881.00000 |      5.00000 |     13.00000 |  83.70000 |               0.43854 |
| transformer |  67905.00000 |     21.00000 |     29.00000 | 357.90000 |               0.40037 |

## 5. Irrigation decision module (threshold proxy)
Threshold: **0.3640 m³/m³** — 25th percentile of training-period 14174 target values. Positive class = IRRIGATE. Reference decisions come from applying the same threshold to observed SM20.

|             |   accuracy |   precision_irrigate |   recall_irrigate |   f1_irrigate |   positive_rate_actual |
|:------------|-----------:|---------------------:|------------------:|--------------:|-----------------------:|
| Persistence |     0.9800 |               0.9893 |            0.9893 |        0.9893 |                 0.9358 |
| LSTM        |     0.9797 |               0.9907 |            0.9876 |        0.9891 |                 0.9358 |
| BiLSTM      |     0.9794 |               0.9896 |            0.9883 |        0.9890 |                 0.9358 |
| Transformer |     0.9800 |               0.9897 |            0.9890 |        0.9893 |                 0.9358 |

Confusion matrices (rows actual NO/YES, columns predicted NO/YES): Persistence [[168, 31], [31, 2870]]; LSTM [[172, 27], [36, 2865]]; BiLSTM [[169, 30], [34, 2867]]; Transformer [[169, 30], [32, 2869]].

Note: the test period is much drier than training, so 93.6% of test hours are IRRIGATE at the default threshold and accuracy is dominated by that class. The sweep below (recall for IRRIGATE at thresholds set from training percentiles) is more informative:

|                          |   BiLSTM |   LSTM |   Persistence |   Transformer |   positive_rate_actual |
|:-------------------------|---------:|-------:|--------------:|--------------:|-----------------------:|
| (5, 0.2730000019073486)  |   0.9764 | 0.9838 |        0.9629 |        0.9710 |                 0.4781 |
| (10, 0.3000000119209289) |   0.9875 | 0.9875 |        0.9823 |        0.9854 |                 0.6187 |
| (15, 0.3179999887943268) |   0.9862 | 0.9849 |        0.9818 |        0.9871 |                 0.7255 |
| (20, 0.3429999947547912) |   0.9901 | 0.9905 |        0.9886 |        0.9916 |                 0.8494 |
| (25, 0.363999992609024)  |   0.9883 | 0.9876 |        0.9893 |        0.9890 |                 0.9358 |
| (30, 0.3729999959468841) |   0.9933 | 0.9933 |        0.9913 |        0.9933 |                 0.9587 |
| (35, 0.3810000121593475) |   0.9944 | 0.9950 |        0.9944 |        0.9940 |                 0.9758 |
| (40, 0.3880000114440918) |   0.9954 | 0.9961 |        0.9947 |        0.9967 |                 0.9823 |
| (45, 0.3970000147819519) |   0.9958 | 0.9967 |        0.9961 |        0.9971 |                 0.9919 |
| (50, 0.4070000052452087) |   0.9987 | 0.9987 |        0.9990 |        0.9987 |                 0.9981 |

## 6. Explainability

Permutation importance (increase in test RMSE, m³/m³):

| feature           |    BiLSTM |     LSTM |   Transformer |
|:------------------|----------:|---------:|--------------:|
| SOIL_MOISTURE_20  |  0.069859 | 0.067836 |      0.070582 |
| P_CALC            |  0.000669 | 0.001591 |      0.001251 |
| SOIL_MOISTURE_10  |  0.000078 | 0.000641 |      0.000062 |
| SUR_TEMP          |  0.000015 | 0.000572 |      0.000017 |
| SOIL_MOISTURE_5   |  0.000012 | 0.000262 |      0.000079 |
| SOLARAD           |  0.000041 | 0.000211 |      0.000029 |
| T_HR_AVG          |  0.000008 | 0.000103 |      0.000029 |
| RH_HR_AVG         |  0.000012 | 0.000034 |      0.000030 |
| SOIL_TEMP_20      |  0.000009 | 0.000033 |      0.000024 |
| SOIL_MOISTURE_100 |  0.000000 | 0.000014 |      0.000025 |
| SOIL_MOISTURE_50  | -0.000000 | 0.000024 |      0.000010 |

Mean |GradientSHAP| (network output, summed over lags):

|                   |     LSTM |   BiLSTM |   Transformer |
|:------------------|---------:|---------:|--------------:|
| T_HR_AVG          | 0.001116 | 0.000449 |      0.000844 |
| P_CALC            | 0.000644 | 0.000441 |      0.000553 |
| SOLARAD           | 0.001447 | 0.000801 |      0.001057 |
| SUR_TEMP          | 0.002036 | 0.000496 |      0.001045 |
| RH_HR_AVG         | 0.001559 | 0.000625 |      0.001109 |
| SOIL_TEMP_20      | 0.000758 | 0.000798 |      0.000576 |
| SOIL_MOISTURE_5   | 0.007605 | 0.002996 |      0.012559 |
| SOIL_MOISTURE_10  | 0.011740 | 0.004339 |      0.009384 |
| SOIL_MOISTURE_20  | 0.068895 | 0.016991 |      0.073395 |
| SOIL_MOISTURE_50  | 0.005759 | 0.001579 |      0.008627 |
| SOIL_MOISTURE_100 | 0.000635 | 0.000532 |      0.000542 |

- BiLSTM: largest permutation importance = soil moisture 20 cm (target history) (+0.0699 m³/m³ RMSE), then precipitation (+0.0007) and soil moisture 10 cm (+0.0001).
- LSTM: largest permutation importance = soil moisture 20 cm (target history) (+0.0678 m³/m³ RMSE), then precipitation (+0.0016) and soil moisture 10 cm (+0.0006).
- Transformer: largest permutation importance = soil moisture 20 cm (target history) (+0.0706 m³/m³ RMSE), then precipitation (+0.0013) and soil moisture 5 cm (+0.0001).
- LSTM (SHAP, drivers of the predicted 1-h change): soil moisture 20 cm (target history), soil moisture 10 cm, soil moisture 5 cm.
- BiLSTM (SHAP, drivers of the predicted 1-h change): soil moisture 20 cm (target history), soil moisture 10 cm, soil moisture 5 cm.
- Transformer (SHAP, drivers of the predicted 1-h change): soil moisture 20 cm (target history), soil moisture 5 cm, soil moisture 10 cm.
- SOIL_MOISTURE_20 dominates permutation importance largely by construction: the forecaster adds the predicted change to the last observed SM20, so permuting SM20 history moves the forecast baseline itself.
- Large mean |SHAP| on SM20 lags (larger than the typical 1-h change) arises from opposite-signed attributions on adjacent lags, i.e. the networks use the recent rate of change of SM20 rather than its absolute level (see the lag heatmaps).
- Precipitation permutation importance by model: BiLSTM +0.00067, LSTM +0.00159, Transformer +0.00125 m³/m³ — rain events drive the sharp wetting jumps that persistence cannot anticipate.
