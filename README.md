# ⚡ Power Demand Prediction Pipeline

A machine learning pipeline to forecast next-hour electricity demand (in MW) using historical power data, weather variables, and macroeconomic indicators. Built with XGBoost and tuned via Optuna, achieving a **MAPE of ~2.89%** on the validation set.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Feature Engineering](#feature-engineering)
- [Model & Hyperparameter Tuning](#model--hyperparameter-tuning)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Files](#output-files)

---

## Overview

This project predicts the **next hour's electricity demand** for a power grid (Bangladesh PGCB data). The pipeline integrates three data sources — operational power demand records, hourly weather observations, and World Bank economic indicators — to build a feature-rich dataset for gradient-boosted tree regression.

The model is trained on data up to end of 2023 and evaluated on 2024 onwards (temporal holdout split).

---

## Project Structure

```
.
├── Power_demand_predict.ipynb   # Exploratory notebook: EDA, tuning, experiments
├── power_demand_pipeline.py     # Production-ready end-to-end pipeline script
├── data/
│   ├── PGCB_date_power_demand.xlsx  # Raw power demand data (PGCB)
│   ├── weather_data.xlsx            # Hourly weather data
│   └── economic_full_1.csv          # World Bank macroeconomic indicators
├── parity_plot.png              # Output: Actual vs. Predicted scatter plot
└── forecast_plot.png            # Output: Time-series forecast on validation set
```

> If you use the `data/` folder, update the path constants in `power_demand_pipeline.py` accordingly:
> ```python
> DEMAND_PATH   = 'data/PGCB_date_power_demand.xlsx'
> WEATHER_PATH  = 'data/weather_data.xlsx'
> ECONOMIC_PATH = 'data/economic_full_1.csv'
> ```

---

## Data Sources

| Source | File | Description |
|---|---|---|
| Power Demand | `PGCB_date_power_demand.xlsx` | Hourly MW demand, generation, load shedding, and import/source breakdown from PGCB |
| Weather | `weather_data.xlsx` | Hourly weather variables (temperature, precipitation, wind speed, sunshine duration, etc.) |
| Economic | `economic_full_1.csv` | Annual World Bank indicators (GDP growth, population, CPI, energy use, etc.) |

### Economic Indicators Used (World Bank Codes)

| Code | Description |
|---|---|
| `NY.GDP.MKTP.KD.ZG` | GDP growth (annual %) |
| `SP.POP.TOTL` | Total population |
| `NV.IND.TOTL.ZS` | Industry value added (% of GDP) |
| `FP.CPI.TOTL.ZG` | Inflation, CPI (annual %) |
| `EG.USE.PCAP.KG.OE` | Energy use per capita (kg oil equivalent) |
| `EG.EGY.PRIM.PP.KD` | Energy intensity (primary energy per PPP GDP) |

---

## Pipeline Walkthrough

The pipeline runs in 6 sequential steps:

### Step 1 — Load Data
Reads all three source files into Pandas DataFrames. The weather file skips the first 3 header rows.

### Step 2 — Clean & Integrate
- **Demand data:** Rounds timestamps to the nearest hour, removes duplicates, sorts by datetime, fills known sparse columns (`india_adani`, `nepal`, `solar`, `wind`) with 0.
- **Outlier removal:** Applies Modified Z-score method (threshold = 3) per column. Outlier values are set to `NaN` and recovered via linear interpolation with forward/backward fill.
- **Weather data:** Sets `time` as the index, deduplicates.
- **Economic data:** Filters to the 6 relevant World Bank indicators, transposes to have years as rows, and fills missing annual values via linear interpolation.
- **Integration:** Demand and weather are joined on the datetime index (left join). Economic data is merged on the year.

### Step 3 — Feature Engineering
See [Feature Engineering](#feature-engineering) section below.

### Step 4 — Train/Validation Split
A strict **temporal split** is used — no random shuffling — to prevent data leakage:
- **Training set:** All data before `2024-01-01`
- **Validation set:** All data from `2024-01-01` onwards

### Step 5 — Training
Trains an `XGBRegressor` with the best hyperparameters found via Optuna (100 trials). See [Model & Hyperparameter Tuning](#model--hyperparameter-tuning).

### Step 6 — Evaluate & Visualize
Computes MAPE and MAE on the validation set, prints feature importances, and saves two diagnostic plots.

---

## Feature Engineering

Low-signal and source-specific columns are dropped first — this includes the individual generation source breakdown (`gas`, `liquid_fuel`, `coal`, `hydro`, `solar`, `wind`), cross-border import columns (`india_adani`, `nepal`, `india_bheramara_hvdc`, `india_tripura`), and redundant weather fields (`temperature_2m`, `wind_direction_10m`, `cloud_cover`, `soil_temperature_0_to_7cm`). This keeps only the variables with meaningful predictive signal. The following features are then constructed:

### Calendar Features

Raw calendar integers (hour, day-of-week, month) are not fed directly to the model because they are ordinal but not truly linear — the model would wrongly treat hour 23 and hour 0 as being far apart when they are actually adjacent. To fix this, they are converted to **cyclic (sine/cosine) encodings**:

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
dayofweek_sin = sin(2π × dayofweek / 7)
month_sin = sin(2π × month / 12)
```

Using both `sin` and `cos` for hour ensures the model can learn the full circular position (a single `sin` is ambiguous — hour 2 and hour 10 share the same sine value). For day-of-week and month, only the sine is used since the main goal is to capture the rough seasonal position rather than exact circular identity.

Two binary flags are also added to capture domain-specific demand spikes:

| Feature | Description |
|---|---|
| `weekend` | 1 if Saturday or Sunday — weekends have distinctly lower industrial demand |
| `is_peak_hour` | 1 if hour ∈ {10, 11, 12, 18, 19, 20} — captures the midday and evening demand peaks typical of this grid |

### Lag Features

Lag features give the model a direct view of recent demand history, which is by far the strongest signal for next-hour prediction (current `demand_mw` alone carries 77% feature importance). Three lags are used to capture different temporal patterns:

| Feature | Shift | What it captures |
|---|---|---|
| `lag_1` | 1 hour back | Immediate momentum — demand changes smoothly hour-to-hour |
| `lag_24` | 24 hours back | Same hour yesterday — captures the strong daily cycle |
| `lag_168` | 168 hours back | Same hour last week — captures the weekly day-of-week pattern |

### Rolling Statistics

Rolling features summarise recent demand behaviour over a window, giving the model context about whether demand has been trending high or low and how volatile it has been:

| Feature | Computation | What it captures |
|---|---|---|
| `roll_mean_24` | Mean of the 24 hours before the current row | Short-term demand level / trend |
| `roll_std_24` | Std dev of the 24 hours before the current row | Short-term demand volatility |

Both are computed on `demand_mw.shift(1)` (i.e. the window ends one hour before the current row) to strictly avoid leakage from the present timestep.

### Leakage Prevention

Since the task is forecasting the *next* hour's demand, care is taken to ensure no future information bleeds into the features:

- **Target:** `demand_mw.shift(-1)` — the demand value one step ahead.
- **`generation_mw`** is shifted forward by 1 hour so it represents generation *before* the prediction window, not concurrent with it.
- **Lag and rolling features** are all anchored to `t-1` or further back, never `t`.
- Rows with `NaN` values (created by shifting at the boundaries) are dropped via `dropna()` after all features are built.

### Top Features by Importance (from notebook)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `demand_mw` (current) | 77.2% |
| 2 | `lag_1` | 13.5% |
| 3 | `lag_24` | 2.0% |
| 4 | `generation_mw` | 1.2% |
| 5 | `hour_cos` | 1.2% |

---

## Model & Hyperparameter Tuning

**Model:** `XGBRegressor` (XGBoost)

**Tuning:** Optuna with 100 trials, minimising MAPE on the validation set.

**Best Hyperparameters (used in pipeline):**

| Parameter | Value |
|---|---|
| `n_estimators` | 931 |
| `max_depth` | 9 |
| `learning_rate` | 0.01054 |
| `subsample` | 0.7322 |
| `colsample_bytree` | 0.8491 |

The search space was: `n_estimators` ∈ [100, 1000], `max_depth` ∈ [3, 10], `learning_rate` ∈ [0.01, 0.3], `subsample` ∈ [0.5, 1.0], `colsample_bytree` ∈ [0.5, 1.0].

---

## Results

| Metric | Value |
|---|---|
| **MAPE** | ~2.89% |
| **MAE** | reported in MW at runtime |

A MAPE of ~2.89% means the model's hourly demand forecasts are off by less than 3% on average — strong performance for a short-term energy forecasting task.

---

## Installation

### Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- openpyxl (for `.xlsx` reading)
- optuna (for tuning, notebook only)

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl optuna
```

---

## Usage

### Run the full pipeline

```bash
python power_demand_pipeline.py
```

Update the file paths inside `__main__` before running:

```python
DEMAND_PATH   = 'path/to/PGCB_date_power_demand.xlsx'
WEATHER_PATH  = 'path/to/weather_data.xlsx'
ECONOMIC_PATH = 'path/to/economic_full_1.csv'
SPLIT_DATE    = '2024-01-01'
```

### Run the notebook

Open `Power_demand_predict.ipynb` in Jupyter or Google Colab. The notebook was originally developed on Colab with a T4 GPU. Run cells top-to-bottom; Optuna tuning runs 100 trials and will take several minutes.

---

## Configuration

| Variable | Location | Description |
|---|---|---|
| `DEMAND_PATH` | `power_demand_pipeline.py` → `__main__` | Path to PGCB demand Excel file |
| `WEATHER_PATH` | `power_demand_pipeline.py` → `__main__` | Path to weather Excel file |
| `ECONOMIC_PATH` | `power_demand_pipeline.py` → `__main__` | Path to economic CSV |
| `SPLIT_DATE` | `power_demand_pipeline.py` → `__main__` | Train/validation cutoff date |
| `BEST_PARAMS` | `power_demand_pipeline.py` → module level | XGBoost hyperparameters (update after re-tuning) |

---

## Output Files

| File | Description |
|---|---|
| `parity_plot.png` | Scatter plot of actual vs. predicted demand on the validation set. Points close to the diagonal red line indicate accurate predictions. |
| `forecast_plot.png` | Time-series line chart showing actual (blue) vs. predicted (red) demand over the full validation period. |
| `data_final.xlsx` | (Notebook only) The fully processed feature matrix exported after feature engineering, useful for inspection or reuse. |
