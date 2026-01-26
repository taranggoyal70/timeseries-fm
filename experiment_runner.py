import pandas as pd
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from chronos_forecaster import ChronosForecaster
from metrics import compare_uv_mv_metrics, print_metrics_comparison
import json

class ExperimentRunner:
    
    def __init__(self, device="cuda"):
        self.fetcher = DataFetcher()
        self.forecaster = ChronosForecaster(device=device)
        self.results = []
    
    def run_single_experiment(self, df, target_column, forecast_date, 
                             history_days, forecast_days):
        forecast_date = pd.to_datetime(forecast_date)
        
        context_start = forecast_date - timedelta(days=history_days)
        context_end = forecast_date
        test_end = forecast_date + timedelta(days=forecast_days)
        
        context_df = df[(df["timestamp"] >= context_start) & 
                       (df["timestamp"] <= context_end)].copy()
        test_df = df[(df["timestamp"] > context_end) & 
                    (df["timestamp"] <= test_end)].copy()
        
        if len(context_df) < 50:
            print(f"Warning: Context too short ({len(context_df)} days), skipping...")
            return None
        
        if len(test_df) < 5:
            print(f"Warning: Test period too short ({len(test_df)} days), skipping...")
            return None
        
        context_df["item_id"] = "series_1"
        test_df["item_id"] = "series_1"
        
        results = self.forecaster.compare_uv_mv(
            context_df, 
            test_df, 
            target_column, 
            len(test_df)
        )
        
        comparison = compare_uv_mv_metrics(results)
        
        comparison.update({
            "forecast_date": forecast_date.strftime("%Y-%m-%d"),
            "history_days": history_days,
            "forecast_days": forecast_days,
            "context_length": len(context_df),
            "test_length": len(test_df)
        })
        
        return comparison
    
    def run_parameter_sweep(self, df, target_columns, 
                           history_multipliers=[0.5, 1, 2, 3],
                           forecast_horizons=[21, 63],
                           start_date="2020-01-01",
                           end_date="2025-09-30",
                           step_months=1):
        base_history = 252
        
        all_results = []
        
        forecast_dates = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=f"{step_months}MS"
        )
        
        total_experiments = (len(target_columns) * len(history_multipliers) * 
                           len(forecast_horizons) * len(forecast_dates))
        
        print(f"\nRunning {total_experiments} experiments...")
        print(f"  Targets: {len(target_columns)}")
        print(f"  History multipliers: {history_multipliers}")
        print(f"  Forecast horizons: {forecast_horizons}")
        print(f"  Forecast dates: {len(forecast_dates)}")
        
        experiment_count = 0
        
        for target in target_columns:
            for alpha in history_multipliers:
                history_days = int(base_history * alpha)
                for m in forecast_horizons:
                    for forecast_date in forecast_dates:
                        experiment_count += 1
                        
                        print(f"\n[{experiment_count}/{total_experiments}] "
                              f"Target: {target}, History: {history_days}d, "
                              f"Forecast: {m}d, Date: {forecast_date.strftime('%Y-%m-%d')}")
                        
                        try:
                            result = self.run_single_experiment(
                                df, target, forecast_date, history_days, m
                            )
                            
                            if result:
                                result["alpha"] = alpha
                                all_results.append(result)
                                print_metrics_comparison(result)
                        
                        except Exception as e:
                            print(f"Error in experiment: {e}")
                            continue
        
        self.results.extend(all_results)
        return all_results
    
    def save_results(self, filename="experiment_results.json"):
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {filename}")
    
    def load_results(self, filename="experiment_results.json"):
        with open(filename, "r") as f:
            self.results = json.load(f)
        print(f"Loaded {len(self.results)} results from {filename}")
        return self.results
    
    def get_summary_stats(self):
        if not self.results:
            print("No results to summarize")
            return None
        
        df = pd.DataFrame(self.results)
        
        summary = {
            "total_experiments": len(df),
            "targets_tested": df["target"].nunique(),
            "mv_wins": (df["mv_better"] == True).sum(),
            "uv_wins": (df["mv_better"] == False).sum(),
            "mv_win_rate": (df["mv_better"] == True).mean() * 100,
            "avg_mape_improvement": df["mape_improvement"].mean(),
            "median_mape_improvement": df["mape_improvement"].median()
        }
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Total Experiments: {summary['total_experiments']}")
        print(f"Targets Tested: {summary['targets_tested']}")
        print(f"MV Wins: {summary['mv_wins']} ({summary['mv_win_rate']:.1f}%)")
        print(f"UV Wins: {summary['uv_wins']}")
        print(f"Avg MAPE Improvement: {summary['avg_mape_improvement']:.2f}%")
        print(f"Median MAPE Improvement: {summary['median_mape_improvement']:.2f}%")
        print("="*60)
        
        by_target = df.groupby("target").agg({
            "mv_better": lambda x: (x == True).mean() * 100,
            "mape_improvement": "mean"
        }).round(2)
        by_target.columns = ["MV Win Rate (%)", "Avg MAPE Improvement (%)"]
        
        print("\nBy Target:")
        print(by_target)
        
        return summary
