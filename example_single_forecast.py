#!/usr/bin/env python3
"""
Example: Single Forecast Comparison (UV vs MV)

This script demonstrates a single experiment as described in README Section 5:
- Date: 03/31/2025
- History: n = 252 days (1 year)
- Forecast: m = 21 days (1 month)
- Dataset: Magnificent-7 stocks
- Target: AAPL

This is useful for:
1. Understanding the basic workflow
2. Testing your setup
3. Quick experimentation
"""

import pandas as pd
from datetime import datetime
from data_loader import DataLoader
from chronos_experiment_runner import ChronosExperimentRunner
from experiment_config import ExperimentConfig


def main():
    print("="*70)
    print("SINGLE FORECAST EXAMPLE: UV vs MV")
    print("="*70)
    
    # Configuration per README Section 5
    target_date = datetime(2025, 3, 31)
    n = 252  # 1 year history
    m = 21   # 1 month forecast
    target_series = 'AAPL'
    
    print(f"\nParameters:")
    print(f"  Target Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"  History Length (n): {n} days")
    print(f"  Forecast Horizon (m): {m} days")
    print(f"  Target Series: {target_series}")
    
    # Step 1: Load or download data
    print("\n" + "-"*70)
    print("STEP 1: Loading Data")
    print("-"*70)
    
    loader = DataLoader()
    
    try:
        # Try to load existing data
        stocks_df = loader.load_data('stocks')
        print("✓ Loaded existing stock data")
    except FileNotFoundError:
        # Download if not exists
        print("Data not found. Downloading...")
        stocks_df = loader.download_stocks()
        print("✓ Downloaded stock data")
    
    print(f"  Data shape: {stocks_df.shape}")
    print(f"  Date range: {stocks_df['timestamp'].min()} to {stocks_df['timestamp'].max()}")
    print(f"  Series: {[col for col in stocks_df.columns if col != 'timestamp']}")
    
    # Step 2: Initialize forecaster
    print("\n" + "-"*70)
    print("STEP 2: Initializing Chronos-2 Model")
    print("-"*70)
    
    config = ExperimentConfig(
        alpha_values=[1.0],  # n = 252
        forecast_horizons=[21],  # m = 21
        model_size='base',
        device='cuda'  # Change to 'cpu' if no GPU
    )
    
    runner = ChronosExperimentRunner(config)
    
    # Step 3: Run single experiment
    print("\n" + "-"*70)
    print("STEP 3: Running UV vs MV Forecast")
    print("-"*70)
    
    result = runner.run_single_experiment(
        df=stocks_df,
        target_date=target_date,
        n=n,
        m=m,
        series_name=target_series,
        dataset_type='stocks'
    )
    
    if result is None:
        print("✗ Experiment failed (insufficient data)")
        return
    
    # Step 4: Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nTarget: {result['series']}")
    print(f"Date: {result['target_date']}")
    print(f"History: {result['n']} days, Forecast: {result['m']} days")
    
    print(f"\n{'Metric':<20} {'Univariate':<15} {'Multivariate':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'RMSE':<20} {result['uv_rmse']:<15.4f} {result['mv_rmse']:<15.4f} {result['rmse_improvement_pct']:<15.2f}%")
    print(f"{'MAPE':<20} {result['uv_mape']:<15.2f}% {result['mv_mape']:<15.2f}% {result['mape_improvement_pct']:<15.2f}%")
    
    print(f"\n{'Winner (MAPE):':<20} {'Multivariate' if result['mv_better_mape'] else 'Univariate'}")
    
    # Show some predictions vs actuals
    print(f"\n{'Day':<10} {'Actual':<15} {'UV Pred':<15} {'MV Pred':<15}")
    print("-"*70)
    for i in range(min(5, len(result['actual_values']))):
        actual = result['actual_values'][i]
        uv_pred = result['uv_predictions'][i]
        mv_pred = result['mv_predictions'][i]
        print(f"{i+1:<10} {actual:<15.2f} {uv_pred:<15.2f} {mv_pred:<15.2f}")
    
    if len(result['actual_values']) > 5:
        print(f"... ({len(result['actual_values']) - 5} more days)")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if result['mv_better_mape']:
        improvement = result['mape_improvement_pct']
        print(f"✓ Multivariate forecasting is BETTER by {improvement:.2f}%")
        print(f"  This suggests that using information from other stocks")
        print(f"  (MSFT, GOOGL, AMZN, META, TSLA, NVDA) helped predict {target_series}.")
    else:
        decline = -result['mape_improvement_pct']
        print(f"✗ Univariate forecasting is BETTER by {decline:.2f}%")
        print(f"  This suggests that other stocks did not provide useful")
        print(f"  information for predicting {target_series} in this case.")
    
    print("\n" + "="*70)
    print("✓ EXAMPLE COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Try different dates or series")
    print("  2. Run quick test: python main.py --quick-test")
    print("  3. Run full experiments: python main.py --dataset stocks")


if __name__ == "__main__":
    main()
