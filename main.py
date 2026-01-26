#!/usr/bin/env python3
"""
Main execution script for Chronos-2 Multivariate Forecasting Experiments
Per GitHub README specifications: https://github.com/taranggoyal70/timeseries-fm

This script runs all experiments as specified in the README:
1. Downloads data (Magnificent-7 stocks, FRED interest rates)
2. Runs UV vs MV forecasting experiments
3. Calculates RMSE and MAPE metrics
4. Saves results and generates visualizations
"""

import argparse
import sys
import os
from datetime import datetime

from data_loader import DataLoader
from chronos_experiment_runner import ChronosExperimentRunner
from experiment_config import ExperimentConfig
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Run Chronos-2 Multivariate Forecasting Experiments'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['stocks', 'rates', 'combined', 'all'],
        default='all',
        help='Dataset to use: stocks (K=7), rates (K=10), combined (K=17), or all'
    )
    
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download data without running experiments'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download and use existing data'
    )
    
    parser.add_argument(
        '--model-size',
        type=str,
        choices=['small', 'base', 'large'],
        default='base',
        help='Chronos-2 model size (default: base)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to run model on (default: cuda)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2000-01-01',
        help='Start date for experiments (default: 2000-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-09-30',
        help='End date for experiments (default: 2025-09-30)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with limited parameters (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CHRONOS-2 MULTIVARIATE FORECASTING EXPERIMENTS")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: chronos-2-{args.model_size}")
    print(f"Device: {args.device}")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print("="*70)
    
    # Step 1: Download Data
    loader = DataLoader()
    
    if not args.skip_download:
        print("\n" + "="*70)
        print("STEP 1: DOWNLOADING DATA")
        print("="*70)
        
        try:
            if args.dataset in ['stocks', 'all']:
                print("\n[1/3] Downloading Magnificent-7 Stocks...")
                stocks_df = loader.download_stocks(args.start_date, args.end_date)
            
            if args.dataset in ['rates', 'all']:
                print("\n[2/3] Downloading FRED Interest Rates...")
                rates_df = loader.download_interest_rates(args.start_date, args.end_date)
            
            if args.dataset in ['combined', 'all']:
                print("\n[3/3] Downloading Combined Dataset...")
                combined_df = loader.download_combined(args.start_date, args.end_date)
            
            print("\n✓ Data download complete!")
            
        except Exception as e:
            print(f"\n✗ Error downloading data: {e}")
            print("Please check your internet connection and try again.")
            sys.exit(1)
    else:
        print("\n⊳ Skipping data download (using existing data)")
    
    if args.download_only:
        print("\n✓ Download complete. Exiting (--download-only flag set)")
        return
    
    # Step 2: Configure Experiments
    print("\n" + "="*70)
    print("STEP 2: CONFIGURING EXPERIMENTS")
    print("="*70)
    
    if args.quick_test:
        print("⊳ Quick test mode enabled (limited parameters)")
        config = ExperimentConfig(
            alpha_values=[1.0],  # Only n=252
            forecast_horizons=[21],  # Only m=21
            start_date='2024-01-01',
            end_date='2024-12-31',
            step_months=3,  # Quarterly instead of monthly
            model_size=args.model_size,
            device=args.device
        )
    else:
        config = ExperimentConfig(
            alpha_values=[0.5, 1.0, 2.0, 3.0],  # n = 126, 252, 504, 756
            forecast_horizons=[21, 63],  # m = 21, 63 days
            start_date=args.start_date,
            end_date=args.end_date,
            step_months=1,  # Monthly rolling forecasts
            model_size=args.model_size,
            device=args.device
        )
    
    print(f"History multipliers (α): {config.alpha_values}")
    print(f"Forecast horizons (m): {config.forecast_horizons}")
    print(f"Rolling forecast period: {config.start_date} to {config.end_date}")
    print(f"Step: {config.step_months} month(s)")
    
    # Step 3: Run Experiments
    print("\n" + "="*70)
    print("STEP 3: RUNNING EXPERIMENTS")
    print("="*70)
    
    runner = ChronosExperimentRunner(config)
    all_results = []
    
    try:
        if args.dataset in ['stocks', 'all']:
            print("\n" + "-"*70)
            print("EXPERIMENT SET 1: MAGNIFICENT-7 STOCKS (K=7)")
            print("-"*70)
            stocks_df = loader.load_data('stocks')
            stocks_results = runner.run_rolling_experiments(
                stocks_df, 
                'stocks',
                series_names=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
            )
            all_results.extend(stocks_results)
        
        if args.dataset in ['rates', 'all']:
            print("\n" + "-"*70)
            print("EXPERIMENT SET 2: FRED INTEREST RATES (K=10)")
            print("-"*70)
            rates_df = loader.load_data('interest_rates')
            rates_series = ['DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 
                          'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
            rates_results = runner.run_rolling_experiments(
                rates_df,
                'rates',
                series_names=rates_series
            )
            all_results.extend(rates_results)
        
        if args.dataset in ['combined', 'all']:
            print("\n" + "-"*70)
            print("EXPERIMENT SET 3: COMBINED STOCKS + RATES (K=17)")
            print("-"*70)
            combined_df = loader.load_data('combined')
            combined_series = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
                             'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 
                             'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
            combined_results = runner.run_rolling_experiments(
                combined_df,
                'combined',
                series_names=combined_series
            )
            all_results.extend(combined_results)
        
    except Exception as e:
        print(f"\n✗ Error running experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Save Results
    print("\n" + "="*70)
    print("STEP 4: SAVING RESULTS")
    print("="*70)
    
    if all_results:
        json_path, csv_path = runner.save_results(all_results)
        
        # Print summary statistics
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        df_results = pd.DataFrame(all_results)
        
        print(f"\nTotal experiments completed: {len(df_results)}")
        print(f"Datasets tested: {df_results['dataset'].nunique()}")
        print(f"Series tested: {df_results['series'].nunique()}")
        
        # Overall MV vs UV comparison
        mv_wins = (df_results['mv_better_mape'] == True).sum()
        total = len(df_results)
        mv_win_rate = (mv_wins / total * 100) if total > 0 else 0
        
        print(f"\nMV vs UV Performance (based on MAPE):")
        print(f"  MV wins: {mv_wins} / {total} ({mv_win_rate:.1f}%)")
        print(f"  UV wins: {total - mv_wins} / {total} ({100 - mv_win_rate:.1f}%)")
        
        avg_mape_improvement = df_results['mape_improvement_pct'].mean()
        print(f"  Average MAPE improvement: {avg_mape_improvement:.2f}%")
        
        # By dataset
        print("\nPerformance by Dataset:")
        for dataset in df_results['dataset'].unique():
            df_subset = df_results[df_results['dataset'] == dataset]
            mv_wins_subset = (df_subset['mv_better_mape'] == True).sum()
            total_subset = len(df_subset)
            win_rate = (mv_wins_subset / total_subset * 100) if total_subset > 0 else 0
            avg_improvement = df_subset['mape_improvement_pct'].mean()
            print(f"  {dataset}: MV wins {mv_wins_subset}/{total_subset} ({win_rate:.1f}%), "
                  f"Avg improvement: {avg_improvement:.2f}%")
        
        print("\n" + "="*70)
        print("✓ EXPERIMENTS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        
    else:
        print("\n⚠ No results to save (no experiments completed)")


if __name__ == "__main__":
    main()
