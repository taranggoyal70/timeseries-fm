"""
Experiment configuration per README specifications.
"""

from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class ExperimentConfig:
    """Configuration for Chronos-2 forecasting experiments per README."""
    
    # History length multipliers (α)
    # n = α * 252 trading days
    alpha_values: List[float] = None
    
    # Forecast horizons (m)
    # 21 days = 1 month, 63 days = 3 months
    forecast_horizons: List[int] = None
    
    # Time period for rolling forecasts
    start_date: str = "2000-01-01"
    end_date: str = "2025-09-30"
    
    # Rolling forecast step (monthly)
    step_months: int = 1
    
    # Model configuration
    model_size: str = "base"  # small, base, or large
    device: str = "cuda"
    
    # Quantile levels for uncertainty
    quantile_levels: List[float] = None
    
    def __post_init__(self):
        """Set defaults per README specifications."""
        if self.alpha_values is None:
            self.alpha_values = [0.5, 1.0, 2.0, 3.0]
        
        if self.forecast_horizons is None:
            self.forecast_horizons = [21, 63]  # 1 month, 3 months
        
        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.5, 0.9]
    
    def get_history_length(self, alpha: float) -> int:
        """Calculate history length n = α * 252."""
        return int(alpha * 252)
    
    def get_all_combinations(self):
        """Get all parameter combinations for experiments."""
        combinations = []
        for alpha in self.alpha_values:
            n = self.get_history_length(alpha)
            for m in self.forecast_horizons:
                combinations.append({
                    'alpha': alpha,
                    'n': n,
                    'm': m
                })
        return combinations


# Default configuration per README
DEFAULT_CONFIG = ExperimentConfig()

# Single test experiment configuration (README Section 5)
TEST_CONFIG = ExperimentConfig(
    alpha_values=[1.0],  # n = 252
    forecast_horizons=[21],  # m = 21
    start_date="2025-03-31",
    end_date="2025-03-31",
    step_months=1
)
