# Google Stock Price EDA and LSTM Forecasting
# Student: Chandru S | Roll No: 24AD020
# Install: pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow
# NOTE: This script downloads current historical data at runtime. Forecasting is educational,
# not investment advice. Use chronological splits to avoid data leakage.

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Download Google/Alphabet Class A historical data
df = yf.download("GOOGL", start="2018-01-01", auto_adjust=True, progress=False)
if df.empty:
    raise RuntimeError("No data downloaded. Check internet access or download GOOGL.csv manually.")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").drop_duplicates(subset=["Date"])
df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

# 2. Feature engineering and EDA
df["Daily_Return"] = df["Close"].pct_change()
df["MA20"] = df["Close"].rolling(20).mean()
df["MA50"] = df["Close"].rolling(50).mean()
print("Shape:", df.shape)
print(df.describe(include="all"))
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Close"], label="Close")
plt.title("GOOGL Closing Price"); plt.xlabel("Date"); plt.ylabel("USD")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df["Daily_Return"].dropna(), bins=50, kde=True)
plt.title("Daily Return Distribution"); plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Close"], label="Close", alpha=.65)
plt.plot(df["Date"], df["MA20"], label="20-day MA")
plt.plot(df["Date"], df["MA50"], label="50-day MA")
plt.legend(); plt.title("Moving Averages"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(data=df[["Daily_Return", "Volume"]].assign(Volume=df["Volume"]/1e7))
plt.title("Daily Returns and Scaled Volume"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[["Open", "High", "Low", "Close", "Volume", "Daily_Return"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation"); plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["Date"], df["Volume"])
plt.title("Trading Volume"); plt.tight_layout(); plt.show()

# 3. Prepare close-price sequences. Split chronologically BEFORE fitting scaler.
values = df[["Close"]].values.astype("float32")
split_train = int(len(values) * 0.70)
split_val = int(len(values) * 0.85)
scaler = MinMaxScaler()
scaler.fit(values[:split_train])
scaled = scaler.transform(values)

LOOKBACK = 60
def make_sequences(data, start, end, lookback=LOOKBACK):
    X, y = [], []
    for i in range(max(start, lookback), end):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(scaled, LOOKBACK, split_train)
X_val, y_val = make_sequences(scaled, split_train, split_val)
X_test, y_test = make_sequences(scaled, split_val, len(scaled))
X_train = X_train.reshape((-1, LOOKBACK, 1))
X_val = X_val.reshape((-1, LOOKBACK, 1))
X_test = X_test.reshape((-1, LOOKBACK, 1))

# 4. Build and train LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=32, shuffle=False,
                    callbacks=[early_stop], verbose=1)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss vs Epoch"); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.legend(); plt.tight_layout(); plt.show()

# 5. Evaluate on held-out test data
pred_scaled = model.predict(X_test, verbose=0)
pred = scaler.inverse_transform(pred_scaled).ravel()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
mape = np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAPE: {mape:.2f}%")

test_dates = df["Date"].iloc[split_val:split_val + len(actual)]
plt.figure(figsize=(12, 5))
plt.plot(test_dates, actual, label="Actual")
plt.plot(test_dates, pred, label="Predicted")
plt.title("GOOGL Test-Set Forecast"); plt.xlabel("Date"); plt.ylabel("USD")
plt.legend(); plt.tight_layout(); plt.show()

# Report actual metrics from this run; do not copy illustrative chart values from a template.
