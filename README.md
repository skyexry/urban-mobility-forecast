# Short-Term Demand Forecasting for CitiBike NYC: A Multi-Model Benchmark

Benchmarking five model architectures for 72-hour ahead, station-level bike-share demand forecasting on 100 NYC CitiBike stations using data from January 2024 through December 2025.

**Key findings:**
- All neural models reduce MAE by 37–39% over the historical average baseline
- LSTM achieves the lowest overall MAE (−39.1% vs HA)
- STGNN with asymmetric log-cosh loss achieves the lowest peak-hour MAE (12.35 trips/hr), outperforming LSTM during morning and evening rush hours
- Adding a weather branch provides no benefit at this data scale

---

## Results

| Model | MAE | RMSE | ΔMAE vs HA | Peak MAE | Off-peak MAE |
|---|---|---|---|---|---|
| Historical Average | 13.18 | 21.20 | — | — | — |
| LSTM | 8.02 | 13.09 | −39.1% | 12.58 | 6.51 |
| TCN-only | 8.28 | 13.84 | −37.2% | 12.83 | 6.76 |
| STGNN (sym) | 8.20 | 13.69 | −37.8% | 12.36 | 6.81 |
| **STGNN (asym)** | **8.18** | **13.42** | **−37.9%** | **12.35** | **6.79** |
| STGNN + Weather | 8.49 | 14.07 | −35.6% | 13.28 | 6.89 |

Peak MAE averaged over all 100 stations during peak hours (7–9 am, 5–7 pm).

---

## Repository Structure

```
urban-mobility-forecast/
├── preprocessing/
│   └── features.py                 # demand matrix, time features, weather fetch, sliding windows
├── model/
│   ├── stgnn.py                    # STGNN with optional weather branch + Transformer decoder
│   ├── stconv.py                   # STGCNBlock with PyG ChebConv
│   └── tcn.py                      # TCN residual blocks
├── notebooks/
│   ├── 01a_preprocess.ipynb        # trip data → hourly demand matrix
│   ├── 01b_station_selection.ipynb # top-100 station selection + graph filter
│   ├── 02_graph_tuning.ipynb       # proximity graph grid search (σ², θ)
│   ├── 03_features_test.ipynb      # feature engineering validation
│   └── 11_train_eval.ipynb         # final training + evaluation (all models)
├── output/                         # saved training notebooks with outputs
├── report.tex                      # full project report (LaTeX)
└── README.md
```

---

## Data

- **Source:** [CitiBike System Data](https://citibikenyc.com/system-data) (trip-level CSV → Parquet)
- **Scope:** 100 highest-demand stations, January 2024 – December 2025 (700 days, 16,808 hourly steps)
- **Split:** 70% train / 15% val / 15% test (chronological)
- **Input window:** 168 hours (1 week) → **Output window:** 72 hours
- **Weather:** [Open-Meteo Archive API](https://open-meteo.com/) — 9 features (temperature, humidity, wind, precipitation)

**Graph construction:** Gaussian kernel on pairwise station distances, σ²=0.0005, sparsity threshold θ=0.90 → 656 directed edges among 100 nodes.

---

## Models

| Model | Description |
|---|---|
| **HA** | Per-station, per-hour-of-day training mean |
| **LSTM** | 2-layer LSTM (64 hidden), demand only, symmetric log-cosh loss |
| **TCN-only** | 4-layer dilated TCN (64ch), demand + time features, symmetric log-cosh loss |
| **STGNN (sym)** | ChebConv (K=3) + demand TCN + time TCN + Transformer decoder, symmetric log-cosh |
| **STGNN (asym)** | Same architecture, asymmetric log-cosh (under-prediction penalty ×2) |
| **STGNN + Weather** | STGNN (asym) + 4th TCN branch for 9-dim weather features |

---

## Setup

```bash
pip install torch torch-geometric pandas numpy scikit-learn openmeteo-requests
```

Training was run on a single NVIDIA RTX 5090 GPU. Notebooks are numbered and can be run in order.

---

## Exploratory Data Analysis

EDA (demand distributions, temporal patterns, station maps) is available at:
[github.com/skyexry/citi-bike](https://github.com/skyexry/citi-bike)

---

## Report

Full write-up in [`report.tex`](report.tex).
