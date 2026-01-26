#!/usr/bin/env python3
"""
Class Example: Magnificent-7 Stock Forecasting (UV vs MV)
Following the professor's teaching approach from:
https://srdas.github.io/NLPBook/Chronos.html#multivariate-stock-price-forecast

This example demonstrates:
1. Downloading Magnificent-7 stocks
2. Preparing data in Chronos-2 format
3. Running multivariate forecast
4. Running univariate forecast for comparison
5. Calculating MAPE and RMSE
"""

import pandas as pd
import yfinance as yf
import torch
from chronos import Chronos2Pipeline
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("MAGNIFICENT-7 STOCK FORECASTING: UV vs MV")
print("Following Class Example Approach")
print("="*70)

# Step 1: Download Magnificent-7 stocks
print("\nStep 1: Downloading Magnificent-7 stocks...")
print("Tickers: GOOGL, AMZN, AAPL, MSFT, META, NVDA, TSLA")

df = yf.download(
    ["GOOGL", "AMZN", "AAPL", "MSFT", "META", "NVDA", "TSLA"],
    start="2023-08-31",
    end="2025-11-01"
)

print(f"Downloaded {len(df)} days of data")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Step 2: Prepare data in Chronos-2 format
print("\nStep 2: Preparing data for Chronos-2...")

# Extract Close prices
df1 = df["Close"].reset_index()
print(f"Length of series = {len(df)}")

# Add item_id (required by Chronos-2)
df1["item_id"] = "D1"

# Convert 'Date' to datetime and set as index
df1['Date'] = pd.to_datetime(df1['Date'])
df1 = df1.set_index('Date')

# Reindex to business day frequency and forward-fill missing values
df1 = df1.asfreq('B').ffill()

# Reset index to make 'Date' a column again (predict_df expects it)
df1 = df1.reset_index()

print("Data prepared:")
print(df1.head())
print(f"\nColumns: {list(df1.columns)}")

# Step 3: Load Chronos-2 model
print("\nStep 3: Loading Chronos-2 model...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"  # Change to "cpu" if no GPU
)
print("✓ Model loaded")

# Step 4: Split data into context and test
print("\nStep 4: Splitting data...")
context_df = df1.iloc[:500]  # Historical data
test_df = df1.iloc[500:]     # Future data for testing
pred_length = len(test_df)

print(f"Context length: {len(context_df)} days")
print(f"Test length: {len(test_df)} days")
print(f"Prediction length: {pred_length} days")

# Step 5: MULTIVARIATE FORECAST
print("\n" + "="*70)
print("MULTIVARIATE FORECAST (using all 7 stocks)")
print("="*70)

# Target: NVDA (we'll predict NVDA using all stocks)
target_stock = "NVDA"
print(f"Target: {target_stock}")
print(f"Covariates: GOOGL, AMZN, AAPL, MSFT, META, TSLA")

# Create future_df (drop only target, keep other stocks)
future_df_mv = test_df.drop(columns=[target_stock])

# Generate multivariate predictions
pred_df_mv = pipeline.predict_df(
    context_df,
    future_df=future_df_mv,
    prediction_length=pred_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="item_id",
    timestamp_column="Date",
    target=target_stock
)

print("✓ Multivariate forecast complete")

# Step 6: UNIVARIATE FORECAST
print("\n" + "="*70)
print("UNIVARIATE FORECAST (using only NVDA)")
print("="*70)

# Keep only target column for univariate
context_df_uv = context_df[["item_id", "Date", target_stock]]
test_df_uv = test_df[["item_id", "Date", target_stock]]
future_df_uv = test_df_uv.drop(columns=target_stock)

# Generate univariate predictions
pred_df_uv = pipeline.predict_df(
    context_df_uv,
    future_df=future_df_uv,
    prediction_length=pred_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="item_id",
    timestamp_column="Date",
    target=target_stock
)

print("✓ Univariate forecast complete")

# Step 7: Calculate Metrics
print("\n" + "="*70)
print("METRICS COMPARISON")
print("="*70)

# Extract ground truth
ts_ground_truth = test_df.set_index("Date")[target_stock]

# Extract predictions
ts_pred_mv = pred_df_mv.set_index("Date")["predictions"]
ts_pred_uv = pred_df_uv.set_index("Date")["predictions"]

# Calculate MAPE (Mean Absolute Percentage Error)
mape_mv = (abs(ts_ground_truth - ts_pred_mv) / ts_ground_truth).mean()
mape_uv = (abs(ts_ground_truth - ts_pred_uv) / ts_ground_truth).mean()

# Calculate RMSE (Root Mean Squared Error)
rmse_mv = ((ts_ground_truth - ts_pred_mv) ** 2).mean() ** 0.5
rmse_uv = ((ts_ground_truth - ts_pred_uv) ** 2).mean() ** 0.5

print(f"\nTarget Stock: {target_stock}")
print(f"Test Period: {pred_length} days")
print()
print(f"{'Method':<20} {'MAPE':<15} {'RMSE':<15}")
print("-"*50)
print(f"{'Multivariate':<20} {mape_mv:<15.6f} {rmse_mv:<15.4f}")
print(f"{'Univariate':<20} {mape_uv:<15.6f} {rmse_uv:<15.4f}")
print()

# Calculate improvement
mape_improvement = ((mape_uv - mape_mv) / mape_uv) * 100
rmse_improvement = ((rmse_uv - rmse_mv) / rmse_uv) * 100

print(f"MAPE Improvement: {mape_improvement:+.2f}%")
print(f"RMSE Improvement: {rmse_improvement:+.2f}%")

if mape_mv < mape_uv:
    print(f"\n✓ Multivariate is BETTER (lower MAPE by {abs(mape_improvement):.2f}%)")
else:
    print(f"\n✗ Univariate is BETTER (lower MAPE by {abs(mape_improvement):.2f}%)")

# Step 8: Visualization
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

# Plot 1: Multivariate Forecast
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ts_context = context_df.set_index("Date")[target_stock]
ts_pred_mv_full = pred_df_mv.set_index("Date")
ts_pred_uv_full = pred_df_uv.set_index("Date")

# Multivariate plot
ts_context.plot(label="Historical Data", color="royalblue", ax=ax1)
ts_ground_truth.plot(label="Actual (Ground Truth)", color="green", ax=ax1)
ts_pred_mv_full["predictions"].plot(label="MV Forecast", color="tomato", ax=ax1)
ax1.fill_between(
    ts_pred_mv_full.index,
    ts_pred_mv_full["0.1"],
    ts_pred_mv_full["0.9"],
    alpha=0.3,
    label="80% Prediction Interval",
    color="orange"
)
ax1.set_title(f"Multivariate {target_stock} Stock Price Forecast (using all 7 stocks)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Stock Price")
ax1.legend()
ax1.grid(True)

# Univariate plot
ts_context.plot(label="Historical Data", color="royalblue", ax=ax2)
ts_ground_truth.plot(label="Actual (Ground Truth)", color="green", ax=ax2)
ts_pred_uv_full["predictions"].plot(label="UV Forecast", color="purple", ax=ax2)
ax2.fill_between(
    ts_pred_uv_full.index,
    ts_pred_uv_full["0.1"],
    ts_pred_uv_full["0.9"],
    alpha=0.3,
    label="80% Prediction Interval",
    color="lavender"
)
ax2.set_title(f"Univariate {target_stock} Stock Price Forecast (using only {target_stock})")
ax2.set_xlabel("Date")
ax2.set_ylabel("Stock Price")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("mag7_forecast_comparison.png", dpi=150, bbox_inches='tight')
print("✓ Plot saved as 'mag7_forecast_comparison.png'")
plt.show()

# Step 9: Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
This example demonstrates the key difference between UV and MV forecasting:

UNIVARIATE (UV):
- Uses only {target_stock} historical prices
- Simpler, faster
- MAPE: {mape_uv:.6f}, RMSE: {rmse_uv:.4f}

MULTIVARIATE (MV):
- Uses all 7 stocks (GOOGL, AMZN, AAPL, MSFT, META, NVDA, TSLA)
- Captures cross-stock relationships
- MAPE: {mape_mv:.6f}, RMSE: {rmse_mv:.4f}

RESULT: {'MV is better!' if mape_mv < mape_uv else 'UV is better!'}

This matches the approach shown in the class notes:
https://srdas.github.io/NLPBook/Chronos.html#multivariate-stock-price-forecast
""")

print("="*70)
print("✓ EXAMPLE COMPLETE")
print("="*70)
