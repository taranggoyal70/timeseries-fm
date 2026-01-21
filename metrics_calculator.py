"""
Metrics Calculator Module for Chronos-2 Experiments

Calculates RMSE and MAPE metrics per README specifications.
"""

import numpy as np


class MetricsCalculator:
    @staticmethod
    def calculate_rmse(actual, predicted):
        """
        Calculate Root Mean Squared Error
        RMSE = [1/m * sum((x-y)^2)]^(1/2)
        """
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))

    @staticmethod
    def calculate_mape(actual, predicted):
        """
        Calculate Mean Absolute Percentage Error
        MAPE = 1/m * sum(|x-y|/|x|) * 100
        """
        mask = actual != 0
        if not mask.any():
            return np.inf
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    @staticmethod
    def calculate_all_metrics(actual, predicted):
        """Calculate both RMSE and MAPE"""
        return {
            'rmse': MetricsCalculator.calculate_rmse(actual, predicted),
            'mape': MetricsCalculator.calculate_mape(actual, predicted)
        }

    @staticmethod
    def compare_uv_mv_metrics(uv_metrics, mv_metrics):
        """Compare univariate vs multivariate forecasting metrics"""
        return {
            'uv_rmse': uv_metrics['rmse'],
            'mv_rmse': mv_metrics['rmse'],
            'rmse_improvement_pct': ((uv_metrics['rmse'] - mv_metrics['rmse']) / uv_metrics['rmse']) * 100,
            'uv_mape': uv_metrics['mape'],
            'mv_mape': mv_metrics['mape'],
            'mape_improvement_pct': ((uv_metrics['mape'] - mv_metrics['mape']) / uv_metrics['mape']) * 100,
            'mv_better_rmse': mv_metrics['rmse'] < uv_metrics['rmse'],
            'mv_better_mape': mv_metrics['mape'] < uv_metrics['mape']
        }


if __name__ == "__main__":
    # Test metrics calculator
    actual = np.array([100, 110, 105, 115, 120])
    predicted_uv = np.array([98, 112, 103, 118, 119])
    predicted_mv = np.array([101, 109, 106, 114, 121])
    
    calc = MetricsCalculator()
    
    uv_metrics = calc.calculate_all_metrics(actual, predicted_uv)
    mv_metrics = calc.calculate_all_metrics(actual, predicted_mv)
    comparison = calc.compare_uv_mv_metrics(uv_metrics, mv_metrics)
    
    print("UV Metrics:", uv_metrics)
    print("MV Metrics:", mv_metrics)
    print("Comparison:", comparison)
