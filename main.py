"""
Main Script for Chronos-2 Multivariate Forecasting Experiments

Research Questions:
1. Do multivariate (MV) methods produce better predictions than univariate (UV) ones 
   when foundation models are used?
2. Is MV forecasting accuracy better for stocks versus interest rates?
3. Is MV forecasting better when both stocks and interest rates are forecast together?
4. Can we build a large-scale "world" forecasting model?

Per README Specifications - Using Chronos-2
"""

import pandas as pd
import torch
from datetime import datetime
from data_loader import DataLoader
from chronos_experiment_runner import ChronosExperimentRunner


def main():
    print("="*60)
    print("CHRONOS-2 MULTIVARIATE FORECASTING EXPERIMENTS")
    print("="*60)
    
    # Check PyTorch and CUDA
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Step 1: Download Data
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING DATA")
    print("="*60)
    
    loader = DataLoader()
    
    stocks_df = loader.download_stocks(start_date="2000-01-01")
    rates_df = loader.download_interest_rates(start_date="2000-01-01")
    combined_df = loader.download_combined(start_date="2000-01-01")
    
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Stocks: {stocks_df.shape}")
    print(f"Rates: {rates_df.shape}")
    print(f"Combined: {combined_df.shape}")
    
    # Step 2: Initialize Chronos-2 Runner
    print("\n" + "="*60)
    print("STEP 2: INITIALIZING CHRONOS-2")
    print("="*60)
    
    runner = ChronosExperimentRunner()
    
    # Step 3: Run Single Test Experiment
    print("\n" + "="*60)
    print("STEP 3: RUNNING TEST EXPERIMENT")
    print("="*60)
    print("Per README Section 5: n=252, m=21, t=03/31/2025")
    
    stocks_df['item_id'] = 'stocks'
    test_date = datetime(2025, 3, 31)
    n, m = 252, 21
    
    print(f"\nTest: n={n}, m={m}, date={test_date.strftime('%Y-%m-%d')}, series=NVDA")
    
    result = runner.run_single_experiment(
        df=stocks_df,
        target_date=test_date,
        n=n,
        m=m,
        series_name='NVDA',
        dataset_type='stocks'
    )
    
    if result:
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        print(f"Dataset: {result['dataset']}")
        print(f"Series: {result['series']}")
        print(f"Target Date: {result['target_date']}")
        print(f"Context Length (n): {result['n']}")
        print(f"Prediction Length (m): {result['m']}")
        print(f"Alpha (n/252): {result['alpha']:.2f}")
        print(f"\nUV RMSE: {result['uv_rmse']:.4f}")
        print(f"MV RMSE: {result['mv_rmse']:.4f}")
        print(f"RMSE Improvement: {result['rmse_improvement_pct']:.2f}%")
        print(f"\nUV MAPE: {result['uv_mape']:.4f}%")
        print(f"MV MAPE: {result['mv_mape']:.4f}%")
        print(f"MAPE Improvement: {result['mape_improvement_pct']:.2f}%")
        print(f"\nMV Better (RMSE): {result['mv_better_rmse']}")
        print(f"MV Better (MAPE): {result['mv_better_mape']}")
    else:
        print("\n✗ Experiment failed")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
