"""
Train a model to predict next-day household power consumption.
Dataset: UCI Household Power Consumption (household_power_consumption.txt)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "household_power_consumption.txt"

# 1. Load and clean -----------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";", na_values="?", low_memory=False)
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
)
df = df.drop(columns=["Date", "Time"]).set_index("datetime")
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()

# 2. Resample to daily ---------------------------------------------------
# Global_active_power is in kW per minute reading; divide the per-day sum
# by 60 to convert to true daily kWh.
daily = (df["Global_active_power"].resample("D").sum() / 60).to_frame()
daily = daily.dropna()

# 3. Feature engineering --------------------------------------------------
daily["day_of_week"] = daily.index.dayofweek
daily["month"] = daily.index.month
daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
daily["lag_1"] = daily["Global_active_power"].shift(1)
daily["lag_7"] = daily["Global_active_power"].shift(7)
daily["rolling_7"] = daily["Global_active_power"].shift(1).rolling(7).mean()
daily = daily.dropna()

FEATURES = ["day_of_week", "month", "is_weekend", "lag_1", "lag_7", "rolling_7"]
TARGET = "Global_active_power"

# 4. Time-based train/test split ------------------------------------------
split_idx = int(len(daily) * 0.85)
train, test = daily.iloc[:split_idx], daily.iloc[split_idx:]

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# 5. Train -----------------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# 6. Evaluate ----------------------------------------------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"MAE:  {mae:.3f} kWh")
print(f"RMSE: {rmse:.3f} kWh")

# 7. Save ----------------------------------------------------------------
joblib.dump(model, PROJECT_DIR / "power_model.pkl")
daily.to_csv(PROJECT_DIR / "daily_power.csv")
print("Saved power_model.pkl and daily_power.csv")
