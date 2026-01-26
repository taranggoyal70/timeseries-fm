"""
Main experiment runner for Chronos-2 forecasting per README specifications.
Implements rolling forecasts with multiple parameter combinations.
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple
import json
import os
from tqdm import tqdm

from chronos import Chronos2Pipeline
from experiment_config import ExperimentConfig
from metrics_calculator import MetricsCalculator


class ChronosExperimentRunner:
    """Run Chronos-2 forecasting experiments per README specifications."""
    
    def __init__(self, config: ExperimentConfig = None):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration (uses default if None)
        """
        self.config = config if config is not None else ExperimentConfig()
        self.metrics_calc = MetricsCalculator()
        self.results = []
        
        # Load Chronos-2 model
        print(f"Loading Chronos-2 {self.config.model_size} model...")
        device = self.config.device if torch.cuda.is_available() else "cpu"
        self.pipeline = Chronos2Pipeline.from_pretrained(
            f"amazon/chronos-2-{self.config.model_size}",
            device_map=device
        )
        print(f"✓ Model loaded on {device}")
    
    def prepare_data_for_date(self, df: pd.DataFrame, target_date: datetime, 
                             n: int, m: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare context and test data for a specific date.
        
        Args:
            df: Full dataset with timestamp column
            target_date: Forecast date t
            n: History length (context)
            m: Forecast horizon
        
        Returns:
            (context_df, test_df) tuple
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Find index of target date
        target_idx = df[df['timestamp'] <= target_date].index[-1]
        
        # Context: n days before target_date
        context_start_idx = max(0, target_idx - n + 1)
        context_df = df.iloc[context_start_idx:target_idx+1].copy()
        
        # Test: m days after target_date
        test_start_idx = target_idx + 1
        test_end_idx = min(len(df), test_start_idx + m)
        test_df = df.iloc[test_start_idx:test_end_idx].copy()
        
        return context_df, test_df
    
    def forecast_univariate(self, context_df: pd.DataFrame, target_column: str, 
                           prediction_length: int) -> pd.DataFrame:
        """
        Univariate forecast for single series.
        
        Args:
            context_df: Historical data with item_id, timestamp, and target column
            target_column: Column to forecast
            prediction_length: Number of steps ahead (m)
        
        Returns:
            DataFrame with predictions
        """
        # Prepare UV context (only target series)
        context_uv = context_df[["item_id", "timestamp", target_column]].copy()
        
        # Create future timestamps
        future_df = pd.DataFrame({
            "item_id": context_uv["item_id"].iloc[-1],
            "timestamp": pd.date_range(
                start=context_uv["timestamp"].iloc[-1] + pd.Timedelta(days=1),
                periods=prediction_length,
                freq="B"
            )
        })
        
        # Forecast
        pred_df = self.pipeline.predict_df(
            context_uv,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=self.config.quantile_levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target=target_column
        )
        
        return pred_df
    
    def forecast_multivariate(self, context_df: pd.DataFrame, target_column: str,
                             prediction_length: int) -> pd.DataFrame:
        """
        Multivariate forecast using all series in context.
        
        Args:
            context_df: Historical data with item_id, timestamp, and all series
            target_column: Column to forecast
            prediction_length: Number of steps ahead (m)
        
        Returns:
            DataFrame with predictions
        """
        # Create future timestamps (NO future covariate values)
        future_df = pd.DataFrame({
            "item_id": context_df["item_id"].iloc[-1],
            "timestamp": pd.date_range(
                start=context_df["timestamp"].iloc[-1] + pd.Timedelta(days=1),
                periods=prediction_length,
                freq="B"
            )
        })
        
        # Forecast using historical relationships
        pred_df = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=self.config.quantile_levels,
            id_column="item_id",
            timestamp_column="timestamp",
            target=target_column
        )
        
        return pred_df
    
    def run_single_experiment(self, df: pd.DataFrame, target_date: datetime,
                             n: int, m: int, series_name: str, 
                             dataset_type: str) -> Dict:
        """
        Run single UV vs MV experiment for one series at one date.
        
        Args:
            df: Full dataset
            target_date: Forecast date t
            n: History length
            m: Forecast horizon
            series_name: Series to forecast
            dataset_type: 'stocks', 'rates', or 'combined'
        
        Returns:
            Dictionary with experiment results
        """
        # Prepare data
        context_df, test_df = self.prepare_data_for_date(df, target_date, n, m)
        
        if len(test_df) < m:
            # Not enough future data
            return None
        
        # Add item_id if not present
        if 'item_id' not in context_df.columns:
            context_df['item_id'] = dataset_type
            test_df['item_id'] = dataset_type
        
        # Run UV forecast
        uv_pred = self.forecast_univariate(context_df, series_name, m)
        
        # Run MV forecast
        mv_pred = self.forecast_multivariate(context_df, series_name, m)
        
        # Get actual values
        actual = test_df.set_index("timestamp")[series_name].values[:m]
        
        # Get median predictions (quantile 0.5)
        uv_values = uv_pred[series_name].values[:m]
        mv_values = mv_pred[series_name].values[:m]
        
        # Calculate metrics
        uv_metrics = self.metrics_calc.calculate_all_metrics(actual, uv_values)
        mv_metrics = self.metrics_calc.calculate_all_metrics(actual, mv_values)
        comparison = self.metrics_calc.compare_uv_mv_metrics(uv_metrics, mv_metrics)
        
        # Store results
        result = {
            'dataset': dataset_type,
            'series': series_name,
            'target_date': target_date.strftime('%Y-%m-%d'),
            'n': n,
            'm': m,
            'alpha': n / 252,
            **comparison,
            'actual_values': actual.tolist(),
            'uv_predictions': uv_values.tolist(),
            'mv_predictions': mv_values.tolist()
        }
        
        return result
    
    def run_rolling_experiments(self, df: pd.DataFrame, dataset_type: str,
                               series_names: List[str] = None) -> List[Dict]:
        """
        Run rolling forecast experiments per README specifications.
        
        Args:
            df: Full dataset
            dataset_type: 'stocks', 'rates', or 'combined'
            series_names: List of series to forecast (all if None)
        
        Returns:
            List of experiment results
        """
        if series_names is None:
            series_names = [col for col in df.columns if col not in ['timestamp', 'item_id']]
        
        # Get all parameter combinations
        combinations = self.config.get_all_combinations()
        
        # Generate forecast dates (monthly rolling)
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        forecast_dates = []
        current_date = start_date
        while current_date <= end_date:
            forecast_dates.append(current_date)
            current_date += relativedelta(months=self.config.step_months)
        
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENTS: {dataset_type.upper()}")
        print(f"{'='*60}")
        print(f"Series: {len(series_names)}")
        print(f"Parameter combinations: {len(combinations)}")
        print(f"Forecast dates: {len(forecast_dates)}")
        print(f"Total experiments: {len(series_names) * len(combinations) * len(forecast_dates)}")
        
        results = []
        
        # Progress bar
        total_experiments = len(series_names) * len(combinations) * len(forecast_dates)
        pbar = tqdm(total=total_experiments, desc="Running experiments")
        
        for series_name in series_names:
            for combo in combinations:
                n, m = combo['n'], combo['m']
                
                for target_date in forecast_dates:
                    # Check if we have enough history
                    df_before_date = df[df['timestamp'] <= target_date]
                    if len(df_before_date) < n:
                        pbar.update(1)
                        continue
                    
                    try:
                        result = self.run_single_experiment(
                            df, target_date, n, m, series_name, dataset_type
                        )
                        
                        if result is not None:
                            results.append(result)
                    
                    except Exception as e:
                        print(f"\n✗ Error in experiment: {series_name}, {target_date}, n={n}, m={m}")
                        print(f"  {str(e)}")
                    
                    pbar.update(1)
        
        pbar.close()
        
        print(f"\n✓ Completed {len(results)} experiments")
        
        return results
    
    def save_results(self, results: List[Dict], output_dir: str = "results"):
        """Save experiment results to JSON and CSV."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        json_path = os.path.join(output_dir, f"experiments_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved full results to {json_path}")
        
        # Save summary as CSV (without actual/predicted values)
        summary_results = []
        for r in results:
            summary = {k: v for k, v in r.items() 
                      if k not in ['actual_values', 'uv_predictions', 'mv_predictions']}
            summary_results.append(summary)
        
        df_summary = pd.DataFrame(summary_results)
        csv_path = os.path.join(output_dir, f"experiments_summary_{timestamp}.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"✓ Saved summary to {csv_path}")
        
        return json_path, csv_path


if __name__ == "__main__":
    print("Chronos-2 Experiment Runner")
    print("Per README specifications")
    print("="*60)
