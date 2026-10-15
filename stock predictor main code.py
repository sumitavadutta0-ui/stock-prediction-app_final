from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Scikit-Learn Machine Learning Models & Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow / Keras for Deep Learning LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# Reproducibility seeds
np.random.seed(42)
tf.random.set_seed(42)

NVIDIA         = "NVDA"            
TESLA          = "TSLA"            
RELIANCE       = "RELIANCE.NS"     
ESCORTS_KUBOTA = "ESCORTS.NS"      
MICROSOFT      = "MSFT"            
GOOGLE         = "GOOGL"           
BOAT_ETF       = "BOAT"            
DOGECOIN       = "DOGE-USD"        

# >>> SELECT YOUR ACTIVE ASSET HERE (e.g., NVIDIA or TESLA) <<<
SELECTED_COMPANY = NVIDIA

# Dynamic 15-Year Time Window ending precisely at today's date
end_date_dt = datetime.today()
start_date_dt = end_date_dt - timedelta(days=int(15 * 365.25))

START_DATE = start_date_dt.strftime("%Y-%m-%d")
END_DATE = end_date_dt.strftime("%Y-%m-%d")
LOOKBACK_DAYS = 60  # Sliding sequence window (60 past trading days)

print(f" RUNNING AI PREDICTOR FOR: {SELECTED_COMPANY}")
print(f" TIME HORIZON: {START_DATE} to {END_DATE} (15+ Years)")


print("\n--- 1: Inspecting Company Profile & Financials ---")
stock_obj = yf.Ticker(SELECTED_COMPANY)

try:
    info = stock_obj.info
    company_name = info.get("longName", info.get("shortName", SELECTED_COMPANY))
    sector = info.get("sector", "N/A")
    currency = info.get("currency", "INR" if ".NS" in SELECTED_COMPANY else "USD")
    current_price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
    market_cap = info.get("marketCap", "N/A")
    
    print(f"Company / Asset Name : {company_name}")
    print(f"Sector               : {sector}")
    print(f"Current Market Price : {current_price} {currency}")
    if isinstance(market_cap, (int, float)):
        print(f"Market Capitalization: {market_cap:,} {currency}")
except Exception as err:
    print(f"[Notice] Fundamental metadata skipped: {err}")
    currency = "INR" if ".NS" in SELECTED_COMPANY else "USD"


print("\n--- 2: Fetching Live Historical Data ---")

df = yf.download(SELECTED_COMPANY, start=START_DATE, end=END_DATE, progress=False)

if df.empty:
    raise ValueError(f"CRITICAL ERROR: Failed to download data for {SELECTED_COMPANY}. Check your internet connection or ticker validity.")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

df.dropna(inplace=True)

if len(df) <= LOOKBACK_DAYS:
    raise ValueError(f"CRITICAL ERROR: Only {len(df)} days found. Need at least {LOOKBACK_DAYS + 1} days to train.")

print(f"Total Trading Sessions Recorded : {len(df)}")


print("\n---  3: Preprocessing & Sequence Engineering ---")

close_data = df['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

X, y = [], []
for i in range(LOOKBACK_DAYS, len(scaled_data)):
    X.append(scaled_data[i - LOOKBACK_DAYS:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Chronological Train-Test Split (80% Train, 20% Test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm  = np.reshape(X_test,  (X_test.shape[0],  X_test.shape[1],  1))

print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")


print("\n---  4: Training All AI & ML Models ---")

print("Training [1/5]: Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Training [2/5]: Decision Tree Regressor...")
dt_model = DecisionTreeRegressor(max_depth=6, random_state=42)
dt_model.fit(X_train, y_train)

print("Training [3/5]: Random Forest Regressor (100 Trees)...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("Training [4/5]: Model Average (Ensemble Voting Regressor)...")
avg_model = VotingRegressor(estimators=[
    ('lr', lr_model),
    ('dt', dt_model),
    ('rf', rf_model)
])
avg_model.fit(X_train, y_train)

print("Training [5/5]: TensorFlow LSTM Neural Network...")
lstm_model = Sequential([
    # The new Input layer fix goes right here:
    Input(shape=(X_train_lstm.shape[1], 1)),
    
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(
    X_train_lstm, y_train,
    batch_size=32,
    epochs=15,
    validation_data=(X_test_lstm, y_test),
    verbose=0
)
print("All models trained successfully.")


print("\n--- 5: Generating Predictions & Computing Metrics ---")

pred_lr_scaled   = lr_model.predict(X_test)
pred_dt_scaled   = dt_model.predict(X_test)
pred_rf_scaled   = rf_model.predict(X_test)
pred_avg_scaled  = avg_model.predict(X_test)
pred_lstm_scaled = lstm_model.predict(X_test_lstm, verbose=0).flatten()

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
pred_lr       = scaler.inverse_transform(pred_lr_scaled.reshape(-1, 1)).flatten()
pred_dt       = scaler.inverse_transform(pred_dt_scaled.reshape(-1, 1)).flatten()
pred_rf       = scaler.inverse_transform(pred_rf_scaled.reshape(-1, 1)).flatten()
pred_avg      = scaler.inverse_transform(pred_avg_scaled.reshape(-1, 1)).flatten()
pred_lstm     = scaler.inverse_transform(pred_lstm_scaled.reshape(-1, 1)).flatten()

evaluation_map = {
    "Linear Regression"       : pred_lr,
    "Decision Tree"           : pred_dt,
    "Random Forest"           : pred_rf,
    "Model Average (Ensemble)": pred_avg,
    "TensorFlow LSTM"         : pred_lstm
}

print("=" * 55)
print(f"{'Model Architecture':<26} | {'MAE (' + currency + ')':<11} | {'RMSE (' + currency + ')':<11}")
print("=" * 55)
for name, preds in evaluation_map.items():
    mae  = mean_absolute_error(y_test_actual, preds)
    rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
    print(f"{name:<26} | {mae:<11.2f} | {rmse:<11.2f}")
print("=" * 55)


# Graph A: 15-Year Historical Macro Trend
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Close'], label='Historical Close Price', color='navy', linewidth=1.2)
plt.title(f"{SELECTED_COMPANY} - 15-Year Macro Price Trajectory ({START_DATE} to {END_DATE})", fontsize=13)
plt.xlabel("Date", fontsize=10)
plt.ylabel(f"Price ({currency})", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Graph B: All Models vs. Real Market Prices (Test Period)
test_dates = df.index[LOOKBACK_DAYS + train_size:]

plt.figure(figsize=(15, 7))
plt.plot(test_dates, y_test_actual, label='Actual Market Price', color='black', linewidth=2.2)
plt.plot(test_dates, pred_lr, label='Linear Regression', linestyle='--', alpha=0.7)
plt.plot(test_dates, pred_dt, label='Decision Tree', linestyle=':', alpha=0.7)
plt.plot(test_dates, pred_rf, label='Random Forest', linestyle='-.', alpha=0.7)
plt.plot(test_dates, pred_avg, label='Model Average (Ensemble)', color='crimson', linewidth=1.8)
plt.plot(test_dates, pred_lstm, label='TensorFlow LSTM', color='purple', linewidth=1.8)

plt.title(f"{SELECTED_COMPANY} - Multi-Model Prediction vs. Actual Market Performance", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=11)
plt.ylabel(f"Price ({currency})", fontsize=11)
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

