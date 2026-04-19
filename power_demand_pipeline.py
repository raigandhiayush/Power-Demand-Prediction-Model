# ============================================================
#  Power Demand Prediction Pipeline (LightGBM Version)
# ============================================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
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

def clean_col_names(df):
    """Standardize column names for consistency and ease of use"""
    print("      → Cleaning and standardizing column names...")
    cols = df.columns
    new_cols = []
    for col in cols:
        # Replace special characters with underscores
        new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        # Replace multiple underscores with single
        new_col = re.sub(r'_{2,}', '_', new_col)
        # Strip leading/trailing underscores
        new_col = new_col.strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    return df


def clean_and_integrate(df, wea, eco):
    print("\n[2/6] Cleaning and integrating data...")

    # --- Demand DataFrame ---
    print("      → Processing demand dataframe...")
    df['datetime'] = df['datetime'].dt.round('h')
    df = df.drop_duplicates()
    df.set_index('datetime', inplace=True)
    df = df.sort_index()

    # Fill known sparse columns with 0 (representing zero generation from these sources when NaN)
    for col in ['india_adani', 'nepal', 'solar', 'wind']:
        df[col] = df[col].fillna(0)

    # Modified Z-score outlier removal + interpolation
    print("      → Removing outliers via Modified Z-score (threshold=3) and interpolating...")
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

    # Clean column names
    df = clean_col_names(df)

    print(f"      ✓ Integrated dataframe shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# STEP 3 — Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(df):
    print("\n[3/6] Engineering features...")
    
    # Drop low-signal / redundant columns
    drop_cols = [
        'remarks', 'temperature_2m_C', 'wind_direction_10m', 'cloud_cover',
        'Industry_including_construction_value_added_of_GDP',
        'soil_temperature_0_to_7cm_C', 'india_bheramara_hvdc',
        'india_tripura', 'india_adani', 'nepal', 'gas', 'liquid_fuel',
        'coal', 'hydro', 'solar', 'wind', 'generation_mw',
        'Inflation_consumer_prices_annual',
        'Energy_use_kg_of_oil_equivalent_per_capita', 'Population_total',
        'GDP_growth_annual',
        'Energy_intensity_level_of_primary_energy_MJ_2021_PPP_GDP'
    ]
    # Only drop columns that exist in the dataframe
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    df = df.drop(cols_to_drop, axis=1)
    print(f"      → Dropped {len(cols_to_drop)} low-signal columns")

    # Calendar features
    print("      → Adding calendar features (hour, day-of-week, month)...")
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.day_of_week
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df = df.drop(['dayofweek', 'month'], axis=1)

    # Cyclic hour encoding
    print("      → Encoding hour cyclically (sin/cos)...")
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_peak_hour'] = df['hour'].isin([10, 11, 12, 18, 19, 20]).astype(int)
    df = df.drop('hour', axis=1)

    # Lag features
    print("      → Creating lag features (1h, 24h, 168h)...")
    df['lag_1'] = df['demand_mw'].shift(1)
    df['lag_24'] = df['demand_mw'].shift(24)
    df['lag_168'] = df['demand_mw'].shift(168)

    # Advanced demand-based features
    print("      → Creating demand trend and volatility features...")
    df['demand_trend'] = (df['demand_mw'].shift(1) - df['demand_mw'].shift(25)) / 24
    df['demand_volatility_24h'] = df['demand_mw'].shift(1).rolling(24).std()
    df['demand_momentum'] = df['demand_mw'].shift(1) - df['demand_mw'].shift(2)

    # EWMA features (exponential weighted moving average)
    print("      → Creating exponential weighted moving average features...")
    df['ewma_4h'] = df['demand_mw'].shift(1).ewm(span=4, adjust=False).mean()
    df['ewma_24h'] = df['demand_mw'].shift(1).ewm(span=24, adjust=False).mean()
    df['roll_std_24'] = df['demand_mw'].shift(1).rolling(24).std()

    # Weather interaction features
    print("      → Creating weather interaction features...")
    df['temp_high'] = (df['apparent_temperature_C'] > 30).astype(int)
    df['temp_low'] = (df['apparent_temperature_C'] < 15).astype(int)
    df['humid_temp'] = df['relative_humidity_2m'] * df['apparent_temperature_C'] / 100
    df['temp_abs_deviation'] = np.abs(df['apparent_temperature_C'] - 25)  # 25°C is neutral

    # Target: next-hour demand
    print("      → Defining target as next-hour demand (shift -1)...")
    df['target'] = df['demand_mw'].shift(-1)

    df = df.dropna()
    print(f"      ✓ Final feature set: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"      ✓ Features ({len(df.columns) - 1}): {list(df.drop(columns=['target']).columns)}")
    return df


# ─────────────────────────────────────────────
# STEP 4 — Train / Validation Split
# ───────────────────���─────────────────────────

def split_data(df, split_date='2024-01-01'):
    print(f"\n[4/6] Splitting data (train: before {split_date} | val: from {split_date})...")

    train_df = df[df.index < split_date]
    val_df = df[df.index >= split_date]

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_val = val_df.drop(columns=['target'])
    y_val = val_df['target']

    print(f"      ✓ Training samples:   {len(X_train):,}")
    print(f"      ✓ Validation samples: {len(X_val):,}")
    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────
# STEP 5 — Hyperparameter Tuning & Training
# ─────────────────────────────────────────────

def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=200):
    """Tune hyperparameters using Optuna for LightGBM"""
    print("\n[5a/6] Tuning hyperparameters via Optuna (200 trials)...")
    
    import optuna
    from optuna.pruners import MedianPruner
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
        preds = model.predict(X_val)

        return mean_absolute_percentage_error(y_val, preds)

    study = optuna.create_study(direction="minimize", pruner=MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"      ✓ Best MAPE (validation): {study.best_value:.2%}")
    print(f"      ✓ Best parameters: {study.best_params}")
    return study.best_params


def train(X_train, y_train, best_params=None):
    """Train LightGBM model with best hyperparameters"""
    print("\n[5b/6] Training LightGBM with optimized parameters...")
    
    if best_params is None:
        best_params = {
                'n_estimators': 1000, 
                'max_depth': 10, 
                'learning_rate': 0.034591170828324636, 
                'subsample': 0.7757346599267606, 
                'colsample_bytree': 0.9572923128668338,
                "random_state": 42,
                "n_jobs": -1,
                'verbose':-1
            }

    
    print(f"      → Parameters:")
    for param, value in best_params.items():
        if param not in ['random_state', 'n_jobs', 'verbose']:
            print(f"           {param:20s}: {value}")

    model = LGBMRegressor(**best_params)
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
    mae = mean_absolute_error(y_val, y_pred)
    print(f"      ✓ MAPE : {mape:.2%}")
    print(f"      ✓ MAE  : {mae:.2f} MW")

    # Feature importance
    print("\n      → Computing feature importances...")
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print("\n      Top 15 Features by Importance:")
    print(importance_df.head(15).to_string(index=False))

    # Parity plot
    print("\n      → Saving parity plot → parity_plot.png")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, y_pred, alpha=0.3, s=10, color='steelblue')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Demand (MW)', fontsize=12)
    plt.ylabel('Predicted Demand (MW)', fontsize=12)
    plt.title('Actual vs. Predicted Demand (Parity Plot)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('parity_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Time-series plot
    print("      → Saving time-series forecast plot → forecast_plot.png")
    plt.figure(figsize=(16, 6))
    
    # Plot actual and predicted
    plt.plot(y_val.index, y_val.values, label='Actual Demand', color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(y_val.index, y_pred, label='Predicted Demand', color='red', linewidth=1.5, alpha=0.8)
    
    plt.title('Actual vs. Predicted Demand Over Time (Validation Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Demand (MW)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('forecast_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("      ✓ All plots saved successfully.")
    return importance_df, mape, mae


# ─────────────────────────────────────────────
# ERROR ANALYSIS (Optional)
# ─────────────────────────────────────────────

def analyze_errors(y_val, y_pred, val_df):
    """Analyze prediction errors by hour, season, and load level"""
    print("\n[OPTIONAL] Error Analysis:")
    
    residuals = np.abs(y_val - y_pred)
    
    # Error by hour
    print("\n      → Analyzing errors by hour of day...")
    hourly_error = pd.DataFrame({
        'hour': val_df.index.hour,
        'error_pct': (residuals / y_val * 100).values
    }).groupby('hour')['error_pct'].agg(['mean', 'std', 'min', 'max'])
    
    worst_hour = hourly_error['mean'].idxmax()
    best_hour = hourly_error['mean'].idxmin()
    print(f"      → Worst performing hour: {worst_hour:02d}:00 (avg error: {hourly_error.loc[worst_hour, 'mean']:.2f}%)")
    print(f"      → Best performing hour: {best_hour:02d}:00 (avg error: {hourly_error.loc[best_hour, 'mean']:.2f}%)")
    
    # Error by month
    print("\n      → Analyzing errors by month...")
    monthly_error = pd.DataFrame({
        'month': val_df.index.month,
        'error_pct': (residuals / y_val * 100).values
    }).groupby('month')['error_pct'].agg(['mean', 'std'])
    
    worst_month = monthly_error['mean'].idxmax()
    best_month = monthly_error['mean'].idxmin()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"      → Worst performing month: {month_names[worst_month-1]} (avg error: {monthly_error.loc[worst_month, 'mean']:.2f}%)")
    print(f"      → Best performing month: {month_names[best_month-1]} (avg error: {monthly_error.loc[best_month, 'mean']:.2f}%)")
    
    return hourly_error, monthly_error


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("   Power Demand Prediction Pipeline (LightGBM - Advanced Features)")
    print("=" * 70)

    # ── Configuration ──────────────────────────────────────────────
    DEMAND_PATH = 'PGCB_date_power_demand.xlsx'
    WEATHER_PATH = 'weather_data.xlsx'
    ECONOMIC_PATH = 'economic_full_1.csv'
    SPLIT_DATE = '2025-01-01'
    
    # Set to True to re-tune hyperparameters, False to use predefined ones
    TUNE_HYPERPARAMETERS = False
    N_TRIALS = 200
    
    # Set to True to run optional error analysis
    RUN_ERROR_ANALYSIS = True
    # ────────────────────────────────────────────────────────────────

    try:
        # Load and prepare data
        df, wea, eco = load_data(DEMAND_PATH, WEATHER_PATH, ECONOMIC_PATH)
        df = clean_and_integrate(df, wea, eco)
        df = engineer_features(df)
        X_train, y_train, X_val, y_val = split_data(df, SPLIT_DATE)

        # Hyperparameter tuning (optional)
        if TUNE_HYPERPARAMETERS:
            best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=N_TRIALS)
        else:
            best_params = None  # Uses default in train()

        # Train model
        model = train(X_train, y_train, best_params)

        # Evaluate and visualize
        importance_df, mape, mae = evaluate_and_visualize(model, X_val, y_val, X_train)

        # Optional: Error analysis
        if RUN_ERROR_ANALYSIS:
            val_df = df[df.index >= SPLIT_DATE]
            hourly_error, monthly_error = analyze_errors(y_val, model.predict(X_val), val_df)

        # Summary
        print("\n" + "=" * 70)
        print("   Pipeline Complete! ✓")
        print("=" * 70)
        print(f"   ✓ Final MAPE  : {mape:.2%}")
        print(f"   ✓ Final MAE   : {mae:.2f} MW")
        print(f"   ✓ Validation samples: {len(X_val):,}")
        print(f"   ✓ Training samples:   {len(X_train):,}")
        print("\n   Output files:")
        print("      • parity_plot.png → Actual vs. Predicted scatter plot")
        print("      • forecast_plot.png → Time-series forecast visualization")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during pipeline execution: {str(e)}")
        print("   Please check your data paths and input files.")
        raise