"""
Error metrics calculation per README formulas.
Implements RMSE and MAPE as specified in the README.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

class MetricsCalculator:
    """Calculate forecast error metrics per README specifications."""
    
    @staticmethod
    def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error per README formula:
        RMSE_i = [1/m * sum((x_{i,t+h} - y_{i,t+h})^2)]^(1/2)
        
        Args:
            actual: Actual values x_{i,t+1}, ..., x_{i,t+m}
            predicted: Predicted values y_{i,t+1}, ..., y_{i,t+m}
        
        Returns:
            RMSE value
        """
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted must have same length")
        
        m = len(actual)
        squared_errors = (actual - predicted) ** 2
        rmse = np.sqrt(np.mean(squared_errors))
        
        return float(rmse)
    
    @staticmethod
    def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error per README formula:
        MAPE = 1/m * sum(|x_{i,t+h} - y_{i,t+h}| / |x_{i,t+h}|)
        
        Args:
            actual: Actual values x_{i,t+1}, ..., x_{i,t+m}
            predicted: Predicted values y_{i,t+1}, ..., y_{i,t+m}
        
        Returns:
            MAPE value (as percentage, e.g., 5.2 means 5.2%)
        """
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted must have same length")
        
        # Avoid division by zero
        mask = actual != 0
        if not mask.any():
            return np.inf
        
        actual_masked = actual[mask]
        predicted_masked = predicted[mask]
        
        m = len(actual_masked)
        absolute_percentage_errors = np.abs((actual_masked - predicted_masked) / actual_masked)
        mape = np.mean(absolute_percentage_errors) * 100  # Convert to percentage
        
        return float(mape)
    
    @staticmethod
    def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate both RMSE and MAPE.
        
        Returns:
            Dictionary with 'rmse' and 'mape' keys
        """
        return {
            'rmse': MetricsCalculator.calculate_rmse(actual, predicted),
            'mape': MetricsCalculator.calculate_mape(actual, predicted)
        }
    
    @staticmethod
    def compare_uv_mv_metrics(uv_metrics: Dict[str, float], 
                              mv_metrics: Dict[str, float]) -> Dict:
        """
        Compare UV vs MV metrics.
        
        Returns:
            Dictionary with comparison results
        """
        rmse_improvement = ((uv_metrics['rmse'] - mv_metrics['rmse']) / uv_metrics['rmse']) * 100
        mape_improvement = ((uv_metrics['mape'] - mv_metrics['mape']) / uv_metrics['mape']) * 100
        
        return {
            'uv_rmse': uv_metrics['rmse'],
            'mv_rmse': mv_metrics['rmse'],
            'rmse_improvement_pct': rmse_improvement,
            'uv_mape': uv_metrics['mape'],
            'mv_mape': mv_metrics['mape'],
            'mape_improvement_pct': mape_improvement,
            'mv_better_rmse': mv_metrics['rmse'] < uv_metrics['rmse'],
            'mv_better_mape': mv_metrics['mape'] < uv_metrics['mape']
        }
    
    @staticmethod
    def calculate_series_metrics(results_dict: Dict, series_names: List[str]) -> pd.DataFrame:
        """
        Calculate metrics for multiple series.
        
        Args:
            results_dict: Dictionary with 'actual', 'uv_predictions', 'mv_predictions'
            series_names: List of series names
        
        Returns:
            DataFrame with metrics for each series
        """
        metrics_list = []
        
        for series_name in series_names:
            actual = results_dict['actual'][series_name].values
            uv_pred = results_dict['uv_predictions'][series_name].values
            mv_pred = results_dict['mv_predictions'][series_name].values
            
            uv_metrics = MetricsCalculator.calculate_all_metrics(actual, uv_pred)
            mv_metrics = MetricsCalculator.calculate_all_metrics(actual, mv_pred)
            comparison = MetricsCalculator.compare_uv_mv_metrics(uv_metrics, mv_metrics)
            
            metrics_list.append({
                'series': series_name,
                **comparison
            })
        
        return pd.DataFrame(metrics_list)


if __name__ == "__main__":
    # Test metrics calculation
    actual = np.array([100, 105, 103, 108, 110])
    uv_pred = np.array([101, 104, 105, 107, 109])
    mv_pred = np.array([100.5, 105.2, 102.8, 108.1, 110.5])
    
    calc = MetricsCalculator()
    
    print("UV Metrics:")
    uv_metrics = calc.calculate_all_metrics(actual, uv_pred)
    print(f"  RMSE: {uv_metrics['rmse']:.4f}")
    print(f"  MAPE: {uv_metrics['mape']:.2f}%")
    
    print("\nMV Metrics:")
    mv_metrics = calc.calculate_all_metrics(actual, mv_pred)
    print(f"  RMSE: {mv_metrics['rmse']:.4f}")
    print(f"  MAPE: {mv_metrics['mape']:.2f}%")
    
    print("\nComparison:")
    comparison = calc.compare_uv_mv_metrics(uv_metrics, mv_metrics)
    print(f"  RMSE Improvement: {comparison['rmse_improvement_pct']:.2f}%")
    print(f"  MAPE Improvement: {comparison['mape_improvement_pct']:.2f}%")
    print(f"  MV Better (MAPE): {comparison['mv_better_mape']}")
