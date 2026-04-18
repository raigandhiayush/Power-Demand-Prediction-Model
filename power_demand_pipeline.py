# ============================================================
#  Power Demand Prediction Pipeline
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

color_pal = sns.color_palette()

# ─────────────────────────────────────────────
# STEP 1 — Load Data
# ─────────────────────────────────────────────

def load_data(demand_path, weather_path, economic_path):
    print("[1/6] Loading data...")

    print("      → Reading power demand Excel file...")
    df = pd.read_excel(demand_path)

    print("      → Reading weather Excel file...")
    wea = pd.read_excel(weather_path, skiprows=3)

    print("      → Reading economic CSV file...")
    eco = pd.read_csv(economic_path)

    print(f"      ✓ Demand data:   {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"      ✓ Weather data:  {wea.shape[0]:,} rows × {wea.shape[1]} columns")
    print(f"      ✓ Economic data: {eco.shape[0]:,} rows × {eco.shape[1]} columns")

    return df, wea, eco


# ─────────────────────────────────────────────
# STEP 2 — Clean & Integrate Data
# ─────────────────────────────────────────────

def clean_and_integrate(df, wea, eco):
    print("\n[2/6] Cleaning and integrating data...")

    # --- Demand DataFrame ---
    print("      → Processing demand dataframe...")
    df['datetime'] = df['datetime'].dt.round('h')
    df = df.drop_duplicates()
    df.set_index('datetime', inplace=True)
    df = df.sort_index()

    # Fill known sparse columns with 0
    for col in ['india_adani', 'nepal', 'solar', 'wind']:
        df[col] = df[col].fillna(0)

    # Modified Z-score outlier removal + interpolation
    print("      → Removing outliers via Modified Z-score and interpolating...")
    for col in df.columns:
        if col != 'remarks':
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            if mad != 0:
                mod_z = 0.6745 * (df[col] - median) / mad
                df.loc[mod_z.abs() > 3, col] = np.nan
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].ffill().bfill()

    # --- Weather DataFrame ---
    print("      → Processing weather dataframe...")
    wea = wea.set_index('time')
    wea = wea.sort_index()
    wea = wea.drop_duplicates()
    wea.index.name = 'datetime'

    # --- Economic DataFrame ---
    print("      → Processing economic dataframe...")
    need = [
        'NY.GDP.MKTP.KD.ZG', 'SP.POP.TOTL', 'NV.IND.TOTL.ZS',
        'FP.CPI.TOTL.ZG', 'EG.USE.PCAP.KG.OE', 'EG.EGY.PRIM.PP.KD'
    ]
    eco = eco[eco['Indicator Code'].isin(need)]
    eco = eco.set_index('Indicator Name')
    eco = eco.drop('Country Name', axis=1)
    eco = eco.drop('Indicator Code', axis=1)
    eco = eco.T
    for col in eco.columns:
        if eco[col].isna().any():
            eco[col].interpolate(method='linear', limit_direction='both')
            eco[col] = eco[col].ffill().bfill()

    # --- Join all three ---
    print("      → Joining demand + weather data (left join on datetime)...")
    df = df.join(wea, how='left')

    print("      → Merging economic indicators on year...")
    df['Year'] = df.index.year
    eco.index = eco.index.astype(int)
    df = pd.merge(df, eco, left_on='Year', right_index=True, how='left')
    df = df.drop(columns=['Year'])
    df.index = pd.to_datetime(df.index)

    print(f"      ✓ Integrated dataframe shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# STEP 3 — Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(df):
    print("\n[3/6] Engineering features...")
    
    # Drop low-signal / redundant columns
    drop_cols = [
        'remarks', 'temperature_2m (°C)', 'wind_direction_10m (°)',
        'cloud_cover (%)', 'Industry (including construction), value added (% of GDP)',
        'soil_temperature_0_to_7cm (°C)', 'india_bheramara_hvdc',
        'india_tripura', 'india_adani', 'nepal', 'gas', 'liquid_fuel',
        'coal', 'hydro', 'solar', 'wind'
    ]
    df = df.drop(drop_cols, axis=1)
    print(f"      → Dropped {len(drop_cols)} low-signal columns")
    # Calendar features
    print("      → Adding calendar features (hour, day-of-week, month, weekend)...")
    df['hour']           = df.index.hour
    df['dayofweek']      = df.index.day_of_week
    df['dayofweek_sin']  = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['month']          = df.index.month
    df['month_sin']      = np.sin(2 * np.pi * df['month'] / 12)
    df['weekend']        = df['dayofweek'].isin([5, 6]).astype(int)
    df = df.drop(['dayofweek', 'month'], axis=1)

    # Cyclic hour encoding
    print("      → Encoding hour cyclically (sin/cos)...")
    df['hour_sin']      = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']      = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_peak_hour']  = df['hour'].isin([10, 11, 12, 18, 19, 20]).astype(int)
    df = df.drop('hour', axis=1)

    # Lag features
    print("      → Creating lag features (1h, 24h, 168h)...")
    df['lag_1']   = df['demand_mw'].shift(1)
    df['lag_24']  = df['demand_mw'].shift(24)
    df['lag_168'] = df['demand_mw'].shift(168)

    # Rolling statistics
    print("      → Creating rolling window features (mean, std over 24h)...")
    df['roll_mean_24'] = df['demand_mw'].shift(1).rolling(24).mean()
    df['roll_std_24']  = df['demand_mw'].shift(1).rolling(24).std()

    # Lag generation to avoid leakage
    df['generation_mw'] = df['generation_mw'].shift(1)

    # Target: next-hour demand
    print("      → Defining target as next-hour demand (shift -1)...")
    df['target'] = df['demand_mw'].shift(-1)

    df = df.dropna()
    print(f"      ✓ Final feature set: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"      ✓ Features: {list(df.drop(columns=['target']).columns)}")
    return df


# ─────────────────────────────────────────────
# STEP 4 — Train / Validation Split
# ─────────────────────────────────────────────

def split_data(df, split_date='2024-01-01'):
    print(f"\n[4/6] Splitting data (train: before {split_date} | val: from {split_date})...")

    train_df = df[df.index < split_date]
    val_df   = df[df.index >= split_date]

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_val   = val_df.drop(columns=['target'])
    y_val   = val_df['target']

    print(f"      ✓ Training samples:   {len(X_train):,}")
    print(f"      ✓ Validation samples: {len(X_val):,}")
    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────
# STEP 5 — Training
# ─────────────────────────────────────────────

BEST_PARAMS = {
    "n_estimators":     931,
    "max_depth":        9,
    "learning_rate":    0.010537231305262573,
    "subsample":        0.7322395062051352,
    "colsample_bytree": 0.8491389433356332,
}

def train(X_train, y_train):
    print("\n[5/6] Training XGBoost with best parameters...")
    print(f"      → Parameters: {BEST_PARAMS}")

    model = XGBRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    print("      ✓ Model training complete")
    return model


# ─────────────────────────────────────────────
# STEP 6 — Evaluate & Visualize
# ─────────────────────────────────────────────

def evaluate_and_visualize(model, X_val, y_val, X_train):
    print("\n[6/6] Evaluating model and generating plots...")

    y_pred = model.predict(X_val)

    mape = mean_absolute_percentage_error(y_val, y_pred)
    mae  = mean_absolute_error(y_val, y_pred)
    print(f"      ✓ MAPE : {mape:.2%}")
    print(f"      ✓ MAE  : {mae:.2f} MW")

    # Feature importance
    print("\n      → Computing feature importances...")
    importance_df = pd.DataFrame({
        'Feature':    X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(importance_df.to_string(index=False))

    # Parity plot
    print("\n      → Saving parity plot → parity_plot.png")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, y_pred, alpha=0.3, s=10)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2)
    plt.xlabel('Actual Demand (MW)')
    plt.ylabel('Predicted Demand (MW)')
    plt.title('Actual vs. Predicted Parity Plot')
    plt.tight_layout()
    plt.savefig('parity_plot.png')
    plt.close()

    # Time-series plot
    print("      → Saving time-series forecast plot → forecast_plot.png")
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=y_val.index, y=y_val,  label='Actual Demand',    color='blue')
    sns.lineplot(x=y_val.index, y=y_pred, label='Predicted Demand', color='red')
    plt.title('Actual vs. Predicted Demand Over Time (Validation Set)')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('forecast_plot.png')
    plt.close()

    print("\n      ✓ All plots saved.")
    return importance_df, mape, mae


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("   Power Demand Prediction Pipeline")
    print("=" * 60)

    # ── Paths (update as needed) ──────────────────────────────
    DEMAND_PATH   = '/home/ayush-raigandhi/Documents/Power predict/PGCB_date_power_demand.xlsx'
    WEATHER_PATH  = '/home/ayush-raigandhi/Documents/Power predict/weather_data.xlsx'
    ECONOMIC_PATH = '/home/ayush-raigandhi/Documents/Power predict/economic_full_1.csv'
    SPLIT_DATE    = '2024-01-01'
    # ─────────────────────────────────────────────────────────

    df, wea, eco         = load_data(DEMAND_PATH, WEATHER_PATH, ECONOMIC_PATH)
    df                   = clean_and_integrate(df, wea, eco)
    df                   = engineer_features(df)
    X_train, y_train, \
    X_val, y_val         = split_data(df, SPLIT_DATE)
    model                = train(X_train, y_train)
    importance_df, \
    mape, mae            = evaluate_and_visualize(model, X_val, y_val, X_train)

    print("\n" + "=" * 60)
    print("   Pipeline complete!")
    print(f"   Final MAPE : {mape:.2%}")
    print(f"   Final MAE  : {mae:.2f} MW")
    print("=" * 60)
