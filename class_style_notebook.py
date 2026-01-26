"""
Chronos-2 Forecasting: Class-Style Example
Following professor's teaching approach from:
https://srdas.github.io/NLPBook/Chronos.html

This notebook-style script demonstrates:
1. Setup and imports
2. Stock price forecasting (Magnificent-7)
3. Interest rate forecasting (FRED)
4. UV vs MV comparison with metrics
"""

# ============================================================================
# SETUP
# ============================================================================

import pandas as pd
import torch
from chronos import Chronos2Pipeline
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

print("Installing/importing required packages...")
print("✓ All packages loaded")

# ============================================================================
# LOAD CHRONOS-2 MODEL
# ============================================================================

print("\nLoading Chronos-2 model...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"  # use "cpu" for CPU inference
)
print("✓ Model loaded successfully")

# ============================================================================
# EXAMPLE 1: MAGNIFICENT-7 STOCKS
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: MAGNIFICENT-7 STOCK FORECASTING")
print("="*70)

# Download stock data
print("\nDownloading Magnificent-7 stocks...")
df = yf.download(
    ["GOOGL", "AMZN", "AAPL", "MSFT", "META", "NVDA", "TSLA"],
    start="2023-08-31",
    end="2025-11-01"
)

print(f"Downloaded {len(df)} days of data")
print("\nFirst few rows:")
print(df.head())

# Prepare data
df1 = df["Close"].reset_index()
print(f"\nLength of series = {len(df)}")

df1["item_id"] = "D1"
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.set_index('Date')
df1 = df1.asfreq('B').ffill()
df1 = df1.reset_index()

print("\nPrepared data:")
print(df1.head())

# Split into context and test
context_df = df1.iloc[:500]
test_df = df1.iloc[500:]
pred_length = len(test_df)

print(f"\nContext: {len(context_df)} days")
print(f"Test: {len(test_df)} days")

# ============================================================================
# MULTIVARIATE FORECAST
# ============================================================================

print("\n" + "-"*70)
print("MULTIVARIATE FORECAST (using all 7 stocks to predict NVDA)")
print("-"*70)

target = "NVDA"
future_df = test_df.drop(columns=[target])

pred_df_mv = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=pred_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="item_id",
    timestamp_column="Date",
    target=target
)

print("✓ Multivariate forecast complete")

# Plot multivariate forecast
ts_context = context_df.set_index("Date")[target]
ts_pred = pred_df_mv.set_index("Date")
ts_ground_truth = test_df.set_index("Date")[target]

plt.figure(figsize=(12, 3))
ts_context.plot(label="historical data", color="royalblue")
ts_ground_truth.plot(label="future data (ground truth)", color="green")
ts_pred["predictions"].plot(label="forecast", color="tomato")
plt.fill_between(
    ts_pred.index,
    ts_pred["0.1"],
    ts_pred["0.9"],
    alpha=0.3,
    label="80% prediction interval",
    color="orange"
)
plt.title(f"Multivariate {target} Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("multivariate_forecast.png", dpi=150)
print("✓ Plot saved as 'multivariate_forecast.png'")
plt.show()

# Calculate metrics
mape_mv = (abs(ts_ground_truth - ts_pred["predictions"]) / ts_ground_truth).mean()
rmse_mv = ((ts_ground_truth - ts_pred["predictions"]) ** 2).mean() ** 0.5

print(f"\nMean Absolute Percentage Error (MAPE) = {mape_mv:.6f}")
print(f"Root Mean Squared Error (RMSE) = {rmse_mv:.4f}")

# ============================================================================
# UNIVARIATE FORECAST (for comparison)
# ============================================================================

print("\n" + "-"*70)
print("UNIVARIATE FORECAST (using only NVDA to predict NVDA)")
print("-"*70)

# Univariate: drop all other stocks
context_df_uv = context_df[["item_id", "Date", target]]
test_df_uv = test_df[["item_id", "Date", target]]
future_df_uv = test_df_uv.drop(columns=target)

pred_df_uv = pipeline.predict_df(
    context_df_uv,
    future_df=future_df_uv,
    prediction_length=pred_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="item_id",
    timestamp_column="Date",
    target=target
)

print("✓ Univariate forecast complete")

# Plot univariate forecast
ts_context = context_df_uv.set_index("Date")[target]
ts_pred_uv = pred_df_uv.set_index("Date")
ts_ground_truth = test_df_uv.set_index("Date")[target]

plt.figure(figsize=(12, 3))
ts_context.plot(label="historical data", color="xkcd:azure")
ts_ground_truth.plot(label="future data (ground truth)", color="xkcd:grass green")
ts_pred_uv["predictions"].plot(label="forecast", color="xkcd:violet")
plt.fill_between(
    ts_pred_uv.index,
    ts_pred_uv["0.1"],
    ts_pred_uv["0.9"],
    alpha=0.7,
    label="80% prediction interval",
    color="xkcd:light lavender"
)
plt.title(f"Univariate {target} Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("univariate_forecast.png", dpi=150)
print("✓ Plot saved as 'univariate_forecast.png'")
plt.show()

# Calculate metrics
mape_uv = (abs(ts_ground_truth - ts_pred_uv["predictions"]) / ts_ground_truth).mean()
rmse_uv = ((ts_ground_truth - ts_pred_uv["predictions"]) ** 2).mean() ** 0.5

print(f"\nMean Absolute Percentage Error (MAPE) = {mape_uv:.6f}")
print(f"Root Mean Squared Error (RMSE) = {rmse_uv:.4f}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*70)
print("UV vs MV COMPARISON")
print("="*70)

print(f"\n{'Method':<20} {'MAPE':<20} {'RMSE':<20}")
print("-"*60)
print(f"{'Multivariate':<20} {mape_mv:<20.6f} {rmse_mv:<20.4f}")
print(f"{'Univariate':<20} {mape_uv:<20.6f} {rmse_uv:<20.4f}")
print("-"*60)

improvement_mape = ((mape_uv - mape_mv) / mape_uv) * 100
improvement_rmse = ((rmse_uv - rmse_mv) / rmse_uv) * 100

print(f"{'Improvement':<20} {improvement_mape:+.2f}% {improvement_rmse:+.2f}%")

if mape_mv < mape_uv:
    print(f"\n✓ MULTIVARIATE is better (MAPE reduced by {abs(improvement_mape):.2f}%)")
else:
    print(f"\n✗ UNIVARIATE is better (MAPE increased by {abs(improvement_mape):.2f}%)")

print("\nObservation:")
print("We can see that the forecast error is", 
      "much lower" if mape_mv < mape_uv else "higher",
      "with multivariate forecasting.")

# ============================================================================
# EXAMPLE 2: INTEREST RATES
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: TREASURY INTEREST RATES")
print("="*70)

# Download 10-year treasury rate from FRED
print("\nDownloading 10-year Treasury rate from FRED...")
df_rates = pd.read_csv(
    'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10'
)

print("Data downloaded:")
print(df_rates.head())

# Prepare for forecasting
df_rates = df_rates.rename(columns={'DATE': 'timestamp', 'DGS10': 'rate'})
df_rates['timestamp'] = pd.to_datetime(df_rates['timestamp'])
df_rates['rate'] = pd.to_numeric(df_rates['rate'], errors='coerce')
df_rates = df_rates.dropna()
df_rates['item_id'] = 'treasury_10y'

print(f"\nTotal observations: {len(df_rates)}")

# Use last 500 days for context, forecast 64 days
context_rates = df_rates.iloc[-564:-64].copy()
test_rates = df_rates.iloc[-64:].copy()
f_len = len(test_rates)

print(f"Context: {len(context_rates)} days")
print(f"Forecast: {f_len} days")

# Forecast
future_rates = test_rates[['item_id', 'timestamp']].copy()

pred_rates = pipeline.predict_df(
    context_rates,
    future_df=future_rates,
    prediction_length=f_len,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column='item_id',
    timestamp_column='timestamp',
    target='rate'
)

print("✓ Interest rate forecast complete")

# Plot
ts_context_rates = context_rates.set_index('timestamp')['rate']
ts_pred_rates = pred_rates.set_index('timestamp')
ts_actual_rates = test_rates.set_index('timestamp')['rate']

plt.figure(figsize=(12, 4))
ts_context_rates.plot(label="historical data", color="royalblue")
ts_actual_rates.plot(label="actual", color="green")
ts_pred_rates["predictions"].plot(label="forecast", color="tomato")
plt.fill_between(
    ts_pred_rates.index,
    ts_pred_rates["0.1"],
    ts_pred_rates["0.9"],
    alpha=0.3,
    label="80% prediction interval",
    color="tomato"
)
plt.title("10-Year Treasury Rate Forecast")
plt.xlabel("Date")
plt.ylabel("Interest Rate (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("treasury_forecast.png", dpi=150)
print("✓ Plot saved as 'treasury_forecast.png'")
plt.show()

# Metrics
mape_rates = (abs(ts_actual_rates - ts_pred_rates["predictions"]) / ts_actual_rates).mean()
rmse_rates = ((ts_actual_rates - ts_pred_rates["predictions"]) ** 2).mean() ** 0.5

print(f"\nMean Absolute Percentage Error (MAPE) = {mape_rates:.6f}")
print(f"Root Mean Squared Error (RMSE) = {rmse_rates:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("""
This script demonstrates Chronos-2 forecasting following the class examples:

1. STOCK FORECASTING (Magnificent-7)
   - Multivariate: Uses all 7 stocks to predict NVDA
   - Univariate: Uses only NVDA to predict NVDA
   - Shows MV can leverage cross-stock relationships

2. INTEREST RATE FORECASTING
   - 10-year Treasury rate from FRED
   - Demonstrates time series with different characteristics

3. KEY METRICS
   - MAPE: Mean Absolute Percentage Error (normalized)
   - RMSE: Root Mean Squared Error (absolute)

Reference: https://srdas.github.io/NLPBook/Chronos.html
""")

print("="*70)
print("✓ ALL EXAMPLES COMPLETE")
print("="*70)
