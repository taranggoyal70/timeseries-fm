"""
Chronos-2 Experiment Runner Module

Runs UV vs MV forecasting experiments using Chronos-2 foundation model.
Per README: Using Chronos-2 for univariate and multivariate forecasting.
"""

import pandas as pd
import torch
from datetime import datetime
from chronos import Chronos2Pipeline
from metrics_calculator import MetricsCalculator


class ChronosExperimentRunner:
    def __init__(self, device="cuda"):
        self.metrics_calc = MetricsCalculator()

        print("Loading Chronos-2 model...")
        device_map = device if torch.cuda.is_available() else "cpu"
        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=device_map
        )
        print(f"✓ Chronos-2 loaded on {device_map}")

    def prepare_data_for_date(self, df, target_date, n, m):
        """
        Prepare context and test data for a specific date
        
        Args:
            df: DataFrame with time series data
            target_date: Date to forecast from
            n: Context length (number of historical days)
            m: Prediction length (number of days to forecast)
        
        Returns:
            context_df: Historical data for model input
            test_df: Actual future data for evaluation
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        target_idx = df[df['timestamp'] <= target_date].index[-1]
        context_start_idx = max(0, target_idx - n + 1)
        context_df = df.iloc[context_start_idx:target_idx+1].copy()

        test_start_idx = target_idx + 1
        test_end_idx = min(len(df), test_start_idx + m)
        test_df = df.iloc[test_start_idx:test_end_idx].copy()

        return context_df, test_df

    def forecast_univariate(self, context_df, target_column, prediction_length):
        """
        UV forecasting per README: single series
        
        Args:
            context_df: Historical data
            target_column: Column to forecast
            prediction_length: Number of steps ahead
        
        Returns:
            Prediction DataFrame
        """
        context_uv = context_df[["item_id", "timestamp", target_column]].copy()

        future_df = pd.DataFrame({
            "item_id": context_uv["item_id"].iloc[-1],
            "timestamp": pd.date_range(
                start=context_uv["timestamp"].iloc[-1] + pd.Timedelta(days=1),
                periods=prediction_length,
                freq="B"
            )
        })

        pred_df = self.pipeline.predict_df(
            context_uv,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column="timestamp",
            target=target_column
        )

        return pred_df

    def forecast_multivariate(self, context_df, target_column, prediction_length):
        """
        MV forecasting per README: multiple series, NO future covariates
        
        Args:
            context_df: Historical data with multiple series
            target_column: Column to forecast
            prediction_length: Number of steps ahead
        
        Returns:
            Prediction DataFrame
        """
        future_df = pd.DataFrame({
            "item_id": context_df["item_id"].iloc[-1],
            "timestamp": pd.date_range(
                start=context_df["timestamp"].iloc[-1] + pd.Timedelta(days=1),
                periods=prediction_length,
                freq="B"
            )
        })

        pred_df = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="item_id",
            timestamp_column="timestamp",
            target=target_column
        )

        return pred_df

    def run_single_experiment(self, df, target_date, n, m, series_name, dataset_type):
        """
        Run single UV vs MV experiment per README
        
        Args:
            df: DataFrame with time series data
            target_date: Date to forecast from
            n: Context length
            m: Prediction length
            series_name: Name of series to forecast
            dataset_type: Type of dataset (stocks/rates/combined)
        
        Returns:
            Dictionary with experiment results
        """
        context_df, test_df = self.prepare_data_for_date(df, target_date, n, m)

        if len(test_df) < m:
            return None

        if 'item_id' not in context_df.columns:
            context_df['item_id'] = dataset_type
            test_df['item_id'] = dataset_type

        try:
            uv_pred = self.forecast_univariate(context_df, series_name, m)
            mv_pred = self.forecast_multivariate(context_df, series_name, m)

            actual = test_df.set_index("timestamp")[series_name].values[:m]
            uv_values = uv_pred[series_name].values[:m]
            mv_values = mv_pred[series_name].values[:m]

            uv_metrics = self.metrics_calc.calculate_all_metrics(actual, uv_values)
            mv_metrics = self.metrics_calc.calculate_all_metrics(actual, mv_values)
            comparison = self.metrics_calc.compare_uv_mv_metrics(uv_metrics, mv_metrics)

            return {
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
        except Exception as e:
            print(f"Error: {e}")
            return None


if __name__ == "__main__":
    print("ChronosExperimentRunner module loaded successfully")
    print("Use this module by importing it in your main script")
