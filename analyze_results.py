#!/usr/bin/env python3
"""
Results Analysis and Visualization Script
Loads experiment results and generates comprehensive visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from pathlib import Path

def load_latest_results(results_dir="results"):
    """Load the most recent experiment results."""
    if not os.path.exists(results_dir):
        print(f"✗ Results directory '{results_dir}' not found")
        return None, None
    
    # Find latest JSON and CSV files
    json_files = sorted(Path(results_dir).glob("experiments_*.json"))
    csv_files = sorted(Path(results_dir).glob("experiments_summary_*.csv"))
    
    if not json_files or not csv_files:
        print("✗ No results files found")
        return None, None
    
    latest_json = str(json_files[-1])
    latest_csv = str(csv_files[-1])
    
    print(f"Loading results from:")
    print(f"  JSON: {latest_json}")
    print(f"  CSV:  {latest_csv}")
    
    # Load data
    with open(latest_json, 'r') as f:
        full_results = json.load(f)
    
    summary_df = pd.read_csv(latest_csv)
    
    return full_results, summary_df

def plot_overall_comparison(df, output_dir="visualizations"):
    """Plot overall UV vs MV comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Win rate pie chart
    mv_wins = (df['mv_better_mape'] == True).sum()
    uv_wins = (df['mv_better_mape'] == False).sum()
    
    axes[0, 0].pie([mv_wins, uv_wins], labels=['MV Wins', 'UV Wins'], 
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('MV vs UV Win Rate (based on MAPE)')
    
    # 2. MAPE improvement distribution
    axes[0, 1].hist(df['mape_improvement_pct'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    axes[0, 1].axvline(x=df['mape_improvement_pct'].mean(), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean: {df["mape_improvement_pct"].mean():.2f}%')
    axes[0, 1].set_xlabel('MAPE Improvement (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of MAPE Improvements')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Performance by dataset
    dataset_stats = df.groupby('dataset').agg({
        'mv_better_mape': lambda x: (x == True).mean() * 100,
        'mape_improvement_pct': 'mean'
    }).round(2)
    
    x = range(len(dataset_stats))
    width = 0.35
    
    axes[1, 0].bar([i - width/2 for i in x], dataset_stats['mv_better_mape'], 
                   width, label='MV Win Rate (%)', color='#2ecc71', alpha=0.7)
    axes[1, 0].bar([i + width/2 for i in x], dataset_stats['mape_improvement_pct'], 
                   width, label='Avg MAPE Improvement (%)', color='#3498db', alpha=0.7)
    axes[1, 0].set_xlabel('Dataset')
    axes[1, 0].set_ylabel('Percentage')
    axes[1, 0].set_title('Performance by Dataset')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(dataset_stats.index)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Performance by forecast horizon
    horizon_stats = df.groupby('m').agg({
        'mv_better_mape': lambda x: (x == True).mean() * 100,
        'mape_improvement_pct': 'mean'
    }).round(2)
    
    x = range(len(horizon_stats))
    axes[1, 1].bar([i - width/2 for i in x], horizon_stats['mv_better_mape'], 
                   width, label='MV Win Rate (%)', color='#2ecc71', alpha=0.7)
    axes[1, 1].bar([i + width/2 for i in x], horizon_stats['mape_improvement_pct'], 
                   width, label='Avg MAPE Improvement (%)', color='#3498db', alpha=0.7)
    axes[1, 1].set_xlabel('Forecast Horizon (days)')
    axes[1, 1].set_ylabel('Percentage')
    axes[1, 1].set_title('Performance by Forecast Horizon')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f'{int(h)} days' for h in horizon_stats.index])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'overall_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_series_performance(df, output_dir="visualizations"):
    """Plot performance by series."""
    os.makedirs(output_dir, exist_ok=True)
    
    series_stats = df.groupby('series').agg({
        'mv_better_mape': lambda x: (x == True).mean() * 100,
        'mape_improvement_pct': 'mean',
        'uv_mape': 'mean',
        'mv_mape': 'mean'
    }).round(2)
    
    series_stats = series_stats.sort_values('mape_improvement_pct', ascending=False)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. MV win rate by series
    colors = ['#2ecc71' if x > 50 else '#e74c3c' for x in series_stats['mv_better_mape']]
    axes[0].barh(range(len(series_stats)), series_stats['mv_better_mape'], color=colors, alpha=0.7)
    axes[0].axvline(x=50, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_yticks(range(len(series_stats)))
    axes[0].set_yticklabels(series_stats.index)
    axes[0].set_xlabel('MV Win Rate (%)')
    axes[0].set_title('MV Win Rate by Series (>50% = MV generally better)')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 2. Average MAPE improvement by series
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in series_stats['mape_improvement_pct']]
    axes[1].barh(range(len(series_stats)), series_stats['mape_improvement_pct'], color=colors, alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_yticks(range(len(series_stats)))
    axes[1].set_yticklabels(series_stats.index)
    axes[1].set_xlabel('Average MAPE Improvement (%)')
    axes[1].set_title('Average MAPE Improvement by Series (positive = MV better)')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'series_performance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_temporal_analysis(df, output_dir="visualizations"):
    """Plot temporal trends."""
    os.makedirs(output_dir, exist_ok=True)
    
    df['target_date'] = pd.to_datetime(df['target_date'])
    df['year'] = df['target_date'].dt.year
    
    yearly_stats = df.groupby('year').agg({
        'mv_better_mape': lambda x: (x == True).mean() * 100,
        'mape_improvement_pct': 'mean'
    }).round(2)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 1. MV win rate over time
    axes[0].plot(yearly_stats.index, yearly_stats['mv_better_mape'], 
                 marker='o', linewidth=2, markersize=8, color='#3498db')
    axes[0].axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% (no advantage)')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('MV Win Rate (%)')
    axes[0].set_title('MV Win Rate Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Average improvement over time
    axes[1].plot(yearly_stats.index, yearly_stats['mape_improvement_pct'], 
                 marker='s', linewidth=2, markersize=8, color='#2ecc71')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No improvement')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Average MAPE Improvement (%)')
    axes[1].set_title('Average MAPE Improvement Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'temporal_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()

def plot_sample_forecasts(full_results, n_samples=3, output_dir="visualizations"):
    """Plot sample forecast comparisons."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select diverse samples
    samples = np.random.choice(len(full_results), min(n_samples, len(full_results)), replace=False)
    
    for idx, sample_idx in enumerate(samples):
        result = full_results[sample_idx]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        actual = np.array(result['actual_values'])
        uv_pred = np.array(result['uv_predictions'])
        mv_pred = np.array(result['mv_predictions'])
        
        x = range(len(actual))
        
        # UV forecast
        axes[0].plot(x, actual, 'o-', label='Actual', color='green', linewidth=2, markersize=6)
        axes[0].plot(x, uv_pred, 's--', label='UV Forecast', color='tomato', linewidth=2, markersize=6)
        axes[0].set_title(f"Univariate Forecast - {result['series']} ({result['target_date']})")
        axes[0].set_xlabel('Days Ahead')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add metrics text
        uv_text = f"UV MAPE: {result['uv_mape']:.2f}%\nUV RMSE: {result['uv_rmse']:.4f}"
        axes[0].text(0.02, 0.98, uv_text, transform=axes[0].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # MV forecast
        axes[1].plot(x, actual, 'o-', label='Actual', color='green', linewidth=2, markersize=6)
        axes[1].plot(x, mv_pred, 's--', label='MV Forecast', color='purple', linewidth=2, markersize=6)
        axes[1].set_title(f"Multivariate Forecast - {result['series']} ({result['target_date']})")
        axes[1].set_xlabel('Days Ahead')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add metrics text
        mv_text = f"MV MAPE: {result['mv_mape']:.2f}%\nMV RMSE: {result['mv_rmse']:.4f}\nImprovement: {result['mape_improvement_pct']:+.2f}%"
        axes[1].text(0.02, 0.98, mv_text, transform=axes[1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, f'sample_forecast_{idx+1}_{result["series"]}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()

def generate_summary_report(df, output_dir="visualizations"):
    """Generate text summary report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report = []
    report.append("="*70)
    report.append("CHRONOS-2 FORECASTING RESULTS SUMMARY")
    report.append("="*70)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-"*70)
    report.append(f"Total experiments: {len(df)}")
    report.append(f"Datasets tested: {df['dataset'].nunique()}")
    report.append(f"Series tested: {df['series'].nunique()}")
    report.append(f"Date range: {df['target_date'].min()} to {df['target_date'].max()}")
    report.append("")
    
    # MV vs UV performance
    mv_wins = (df['mv_better_mape'] == True).sum()
    total = len(df)
    mv_win_rate = (mv_wins / total * 100) if total > 0 else 0
    
    report.append("MV vs UV PERFORMANCE (based on MAPE)")
    report.append("-"*70)
    report.append(f"MV wins: {mv_wins} / {total} ({mv_win_rate:.1f}%)")
    report.append(f"UV wins: {total - mv_wins} / {total} ({100 - mv_win_rate:.1f}%)")
    report.append(f"Average MAPE improvement: {df['mape_improvement_pct'].mean():.2f}%")
    report.append(f"Median MAPE improvement: {df['mape_improvement_pct'].median():.2f}%")
    report.append("")
    
    # By dataset
    report.append("PERFORMANCE BY DATASET")
    report.append("-"*70)
    for dataset in df['dataset'].unique():
        df_subset = df[df['dataset'] == dataset]
        mv_wins_subset = (df_subset['mv_better_mape'] == True).sum()
        total_subset = len(df_subset)
        win_rate = (mv_wins_subset / total_subset * 100) if total_subset > 0 else 0
        avg_improvement = df_subset['mape_improvement_pct'].mean()
        report.append(f"{dataset}:")
        report.append(f"  MV wins: {mv_wins_subset}/{total_subset} ({win_rate:.1f}%)")
        report.append(f"  Avg improvement: {avg_improvement:+.2f}%")
    report.append("")
    
    # Top performers
    report.append("TOP 5 SERIES (by MV improvement)")
    report.append("-"*70)
    top_series = df.groupby('series')['mape_improvement_pct'].mean().sort_values(ascending=False).head(5)
    for series, improvement in top_series.items():
        report.append(f"{series}: {improvement:+.2f}%")
    report.append("")
    
    # Bottom performers
    report.append("BOTTOM 5 SERIES (by MV improvement)")
    report.append("-"*70)
    bottom_series = df.groupby('series')['mape_improvement_pct'].mean().sort_values().head(5)
    for series, improvement in bottom_series.items():
        report.append(f"{series}: {improvement:+.2f}%")
    report.append("")
    
    report.append("="*70)
    
    # Save report
    filepath = os.path.join(output_dir, 'summary_report.txt')
    with open(filepath, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved: {filepath}")
    
    # Also print to console
    print("\n" + '\n'.join(report))

def main():
    print("="*70)
    print("CHRONOS-2 RESULTS ANALYSIS & VISUALIZATION")
    print("="*70)
    print()
    
    # Load results
    full_results, summary_df = load_latest_results()
    
    if full_results is None or summary_df is None:
        print("\n✗ No results to analyze. Run experiments first:")
        print("  python main.py --quick-test --dataset stocks")
        sys.exit(1)
    
    print(f"\nLoaded {len(full_results)} experiments")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    print("-"*70)
    
    plot_overall_comparison(summary_df)
    plot_series_performance(summary_df)
    plot_temporal_analysis(summary_df)
    plot_sample_forecasts(full_results, n_samples=3)
    generate_summary_report(summary_df)
    
    print()
    print("="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files in 'visualizations/' directory:")
    print("  - overall_comparison.png")
    print("  - series_performance.png")
    print("  - temporal_analysis.png")
    print("  - sample_forecast_*.png")
    print("  - summary_report.txt")
    print()

if __name__ == "__main__":
    main()
